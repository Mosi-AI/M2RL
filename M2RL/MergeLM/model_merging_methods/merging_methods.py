from collections import defaultdict, OrderedDict
from tqdm import tqdm
import copy
import torch
import torch.nn as nn

from model_merging_methods.task_vector import TaskVector
from utils.utils import get_param_names_to_merge, get_modules_to_merge
from model_merging_methods.mask_weights_utils import mask_model_weights


class MergingMethod:
    def __init__(self, merging_method_name: str):
        """
        Methods for model merging.
        :param merging_method_name: str, name of the merging method, can be "average_merging", "task_arithmetic",
        "fisher_merging", "regmean_merging", "ties_merging", "latent_merging"
        :return:
        """
        self.merging_method_name = merging_method_name

    def copy_params_to_model(self, params: dict, model: nn.Module):
        """
        copy parameters in "params" to the model
        :param params: dict, dictionary of parameters
        :param model: nn.Module, model that needs to copy parameters
        :return:
        """
        for param_name, param_value in model.named_parameters():
            if param_name in params:
                param_value.data.copy_(params[param_name])

    def average_merging(self, models_to_merge: list, exclude_param_names_regex: list):
        """
        average merging method
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :return:
        """
        # dictionary of list, where key is the parameter name,
        # value is a list of the corresponding parameters of all the models that need to be merged
        models_to_merge_param_dict = defaultdict(list)
        # iterate each individual model that needs to be merged
        for model_to_merge in models_to_merge:
            param_dict = {param_name: param_value for param_name, param_value in model_to_merge.named_parameters()}
            # exclude parameter whose name matches element in exclude_param_names_regex
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(param_dict[param_name])

        with torch.no_grad():
            # average merging of individual models' parameters
            averaged_params = {param_name: torch.stack(model_to_merge_param, dim=0).mean(dim=0) for param_name, model_to_merge_param in models_to_merge_param_dict.items()}

        return averaged_params

    def task_arithmetic(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
        """
        task arithmetic method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """
        assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

        # iterate each individual model that needs to be merged
        with torch.no_grad():
            # sum up the task vectors
            merged_task_vector = models_to_merge_task_vectors[0] + models_to_merge_task_vectors[1]
            for index in range(2, len(models_to_merge_task_vectors)):
                merged_task_vector = merged_task_vector + models_to_merge_task_vectors[index]

            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

        return merged_params

    def fisher_merging(self, models_to_merge: list, trainers: list, exclude_param_names_regex: list, nums_fisher_examples: list, fisher_scaling_coefficients: list = None,
                       normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6):
        """
        fisher merging method
        :param models_to_merge: list, individual models that need to be merged
        :param trainers: list, trainers of individual models
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param nums_fisher_examples: list, numbers of examples to compute fisher weights
        :param fisher_scaling_coefficients: list, scaling coefficients to merge fisher weights
        :param normalize_fisher_weight: boolean, whether to normalize fisher weights (L2 norm) or not
        :param minimal_fisher_weight: float, the minimal value in fisher weights, used for tackling the potential numerical issues
        :return:
        """
        def get_param_squared_gradients(model: nn.Module, param_names_to_merge: list):
            """
            get the squared gradients of parameters
            :param model: nn.Module, model
            :param param_names_to_merge: list, list of parameter names that need to be merged
            :return:
            """
            param_squared_gradients = {param_name: param_value.grad.detach() ** 2 for param_name, param_value in model.named_parameters() if param_name in param_names_to_merge}
            return param_squared_gradients

        def get_models_fisher_norm(models_to_merge_param_dict: dict, models_to_merge_fisher_weights_list: list):
            """
            get normalization of fisher weights of all the models that need to be merged
            :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
            value is a list of the corresponding parameters of all the models that need to be merged
            :param models_to_merge_fisher_weights_list: list, list of dictionaries with length len(models_to_merge),
            each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
            :return:
            """
            # dict, key is parameter name, value is a Tensor with shape (num_models_to_merge, )
            models_fisher_norm_dict = {}
            # compute L2 norm over models for each parameter
            for param_name, _ in models_to_merge_param_dict.items():
                # Tensor, shape (num_models_to_merge, *fisher_weight_shape)
                models_fisher = torch.stack([model_to_merge_fisher_weights[param_name] for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list], dim=0)
                dims = [dim_idx for dim_idx in range(1, models_fisher.dim())]
                # Tensor, shape (num_models_to_merge, ), compute L2 norm for each parameter
                models_fisher_norm = torch.norm(models_fisher, dim=dims)
                models_fisher_norm_dict[param_name] = models_fisher_norm

            # Tensor, shape (num_models_to_merge, num_parameters)
            models_fisher_norm = torch.stack([models_fisher_norm for models_fisher_norm in models_fisher_norm_dict.values()], dim=1)
            # Tensor, shape (num_models_to_merge, ), compute L2 norm over all the parameters
            models_fisher_norm = torch.norm(models_fisher_norm, dim=1)
            return models_fisher_norm

        def merging_with_fisher_weights(models_to_merge_param_dict: dict, models_to_merge_fisher_weights_list: list, fisher_scaling_coefficients: torch.Tensor,
                                        normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6):
            """
            merge parameters of different models with computed fisher weights
            :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
            value is a list of the corresponding parameters of all the models that need to be merged
            :param models_to_merge_fisher_weights_list: list, list of dictionaries with length len(models_to_merge),
            each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
            :param fisher_scaling_coefficients: torch.Tensor, scaling coefficients to merge fisher weights
            :param normalize_fisher_weight: boolean, whether to normalize fisher weights (L2 norm) or not
            :param minimal_fisher_weight: float, the minimal value in fisher weights, used for tackling the potential numerical issues
            :return:
            """
            # dict, dictionary of model parameters
            merged_params = {}

            if normalize_fisher_weight:
                # Tensor, shape (num_models_to_merge, ), L2 norm over all the parameters of models that need to be merged
                models_fisher_norm = get_models_fisher_norm(models_to_merge_param_dict=models_to_merge_param_dict,
                                                            models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list)

            for param_name, param_value_list in models_to_merge_param_dict.items():
                # shape (num_models_to_merge, *parameter_shape)
                param_values = torch.stack(param_value_list, dim=0)
                # Tensor, shape (num_models_to_merge, *fisher_weight_shape), use minimal_fisher_weight to solve the potential numerical issues
                models_to_merge_fisher_weights = torch.stack([model_to_merge_fisher_weights[param_name]
                                                              for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list], dim=0) + minimal_fisher_weight

                # Tensor, shape (num_models_to_merge, 1, 1, ...)
                reshaped_scaling_coefficients = fisher_scaling_coefficients.reshape(-1, *[1 for _ in range(param_values.dim() - 1)]).to(param_values.device)

                if normalize_fisher_weight:
                    # Tensor, shape (num_models_to_merge, )
                    _models_fisher_norm = 1.0 / (models_fisher_norm + minimal_fisher_weight)
                    normalized_models_fisher_norm = _models_fisher_norm / _models_fisher_norm.sum()
                    normalized_models_fisher_norm = normalized_models_fisher_norm.reshape(-1, *[1 for _ in range(param_values.dim() - 1)])
                    reshaped_scaling_coefficients = reshaped_scaling_coefficients * normalized_models_fisher_norm

                # shape (*parameter_shape)
                numerator = (reshaped_scaling_coefficients * models_to_merge_fisher_weights * param_values).sum(dim=0)

                # shape (*parameter_shape)
                denominator = (reshaped_scaling_coefficients * models_to_merge_fisher_weights).sum(dim=0)

                merged_param = numerator / denominator
                merged_params[param_name] = merged_param
            return merged_params

        # dictionary of list, where key is the parameter name,
        # value is a list of the corresponding parameters of all the models that need to be merged
        models_to_merge_param_dict = defaultdict(list)

        # list of dictionaries with length len(models_to_merge),
        # each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
        models_to_merge_fisher_weights_list = []

        assert len(models_to_merge) == len(trainers) == len(nums_fisher_examples), "sizes of lists are not identical!"

        for model_idx, (model_to_merge, trainer, num_fisher_examples) in enumerate(zip(models_to_merge, trainers, nums_fisher_examples)):
            param_dict = {param_name: param_value for param_name, param_value in model_to_merge.named_parameters()}
            # exclude parameter whose name matches element in exclude_param_names_regex
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)

            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(param_dict[param_name])

            # list of dictionaries with length (num_fisher_examples // batch_size) or (num_fisher_examples // batch_size) + 1,
            # each dictionary records the fisher weights of parameters for model_to_merge computed by examples in a batch
            batches_fisher_weights_list = []

            num_computed_examples = 0
            train_dataloader = trainer.get_train_dataloader()
            if num_fisher_examples % trainer._train_batch_size != 0:
                print(f"warning: the number of examples for computing fisher cannot be fully divided by the batch size for model {model_idx}, "
                      "which may lead to a slightly different number of the actually used examples.")
            for step, inputs in tqdm(enumerate(train_dataloader), desc=f"computing fisher weights for model {model_idx}"):
                if num_computed_examples >= num_fisher_examples:
                    break
                inputs = trainer._prepare_inputs(inputs)
                outputs = model_to_merge(**inputs)
                # Tensor, shape (batch_size, num_label_classes)
                logits = outputs.logits
                # compute fisher weights for regression task
                if logits.shape[-1] == 1:
                    # use the label information to compute loss and obtain gradients
                    mse_loss = outputs.loss
                    model_to_merge.zero_grad()
                    mse_loss.backward()
                    # dict, fisher weights of a batch
                    batch_fisher_weights = get_param_squared_gradients(model=model_to_merge, param_names_to_merge=param_names_to_merge)
                # compute fisher weights for classification task
                else:
                    # use detach() to detach from the computation graph
                    # Tensor, shape (batch_size, num_label_classes)
                    labels_probabilities = torch.softmax(logits, dim=-1).detach()
                    labels_log_probabilities = torch.log_softmax(logits, dim=-1)
                    # sqrt labels_probabilities, since torch.sqrt(labels_probabilities) would be squared in the following squared gradients
                    labels_expectations = torch.sqrt(labels_probabilities) * labels_log_probabilities
                    # sum over label classes and batch dimension
                    sum_labels_expectations = labels_expectations.sum(dim=-1).sum(dim=0)
                    model_to_merge.zero_grad()
                    sum_labels_expectations.backward()
                    # dict, fisher weights of a batch
                    batch_fisher_weights = get_param_squared_gradients(model=model_to_merge, param_names_to_merge=param_names_to_merge)

                batches_fisher_weights_list.append(batch_fisher_weights)
                num_computed_examples += trainer._train_batch_size

            model_to_merge_fisher_weights = {}
            for batch_fisher_weights in batches_fisher_weights_list:
                for key in batch_fisher_weights:
                    if key not in model_to_merge_fisher_weights:
                        model_to_merge_fisher_weights[key] = batch_fisher_weights[key]
                    else:
                        model_to_merge_fisher_weights[key] += batch_fisher_weights[key]

            # mean over batches
            for key in model_to_merge_fisher_weights:
                model_to_merge_fisher_weights[key] /= num_computed_examples
            models_to_merge_fisher_weights_list.append(model_to_merge_fisher_weights)

        # merging with fisher weights
        # if fisher_scaling_coefficients is None, then set the fisher weights of different models to contribute equally
        if fisher_scaling_coefficients is None:
            fisher_scaling_coefficients = torch.ones(len(models_to_merge)) / len(models_to_merge)
        else:
            assert isinstance(fisher_scaling_coefficients, list), "wrong type of fisher_scaling_coefficients, should be list!"
            assert len(fisher_scaling_coefficients) == len(models_to_merge), "mismatched length of fisher_scaling_coefficients!"
            fisher_scaling_coefficients = torch.Tensor(fisher_scaling_coefficients)
        # merging with fisher weights
        merged_params = merging_with_fisher_weights(models_to_merge_param_dict=models_to_merge_param_dict, models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list,
                                                    fisher_scaling_coefficients=fisher_scaling_coefficients, normalize_fisher_weight=normalize_fisher_weight, minimal_fisher_weight=minimal_fisher_weight)

        return merged_params

    def regmean_merging(self, models_to_merge: list, trainers: list, exclude_param_names_regex: list, nums_regmean_examples: list, reduce_non_diagonal_ratio: float = 1.0):
        """
        regmean merging method
        :param models_to_merge: list, individual models that need to be merged
        :param trainers: list, trainers of individual models
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param nums_regmean_examples: list, numbers of examples to compute regmean weights
        :param reduce_non_diagonal_ratio: float, reduce non-diagonal elements in regmean weights by multiplying this scalar
        :return:
        """
        def compute_regmean_weights(module_name: str):
            """
            compute the regmean weights, a hook function to deal with each module's input
            :param module_name: str, module name
            :return:
            """
            def hook(module: nn.Module, input: tuple, output: torch.Tensor):
                # Tensor, shape (batch_size, sequence_length, hidden_dim)
                x = input[0].detach()
                batch_num_actual_examples = x.shape[0]
                # Tensor, shape (batch_size * sequence_length, hidden_dim)
                x = x.reshape(-1, x.shape[-1])
                # Tensor, shape (hidden_dim, hidden_dim)
                xtx = torch.matmul(x.transpose(0, 1), x)
                # store the averaged weights in regmean_weights
                if module_name not in regmean_weights.keys():
                    regmean_weights[module_name] = xtx / x.shape[0]
                    num_computed_examples[module_name] = x.shape[0]
                    num_actual_examples[module_name] = batch_num_actual_examples
                else:
                    regmean_weights[module_name] = (regmean_weights[module_name] * num_computed_examples[module_name] + xtx) / (num_computed_examples[module_name] + x.shape[0])
                    num_computed_examples[module_name] += x.shape[0]
                    num_actual_examples[module_name] += batch_num_actual_examples
            return hook

        def reduce_non_diagonal_elements(regmean_weights: torch.Tensor, reduce_non_diagonal_ratio: float):
            """
            reduce the non-diagonal elements in regmean_weights
            :param regmean_weights: Tensor, shape (hidden_dim, hidden_dim), input regmean weights
            :param reduce_non_diagonal_ratio: float, reduce non-diagonal elements in regmean weights by multiplying this scalar
            :return:
            """
            # diagonal matrix with (1 - reduce_non_diagonal_ratio) as elements
            diag_weights = torch.diag(torch.ones(regmean_weights.shape[0]) - reduce_non_diagonal_ratio).to(regmean_weights.device)
            # matrix with reduce_non_diagonal_ratio as elements
            non_diag_weights = torch.zeros_like(diag_weights).fill_(reduce_non_diagonal_ratio)
            # diagonal elements are unchanged, while non-diagonal elements are multiplied by reduce_non_diagonal_ratio
            return regmean_weights * (diag_weights + non_diag_weights)

        def merging_with_regmean_weights(models_to_merge_param_dict: dict, models_to_merge_regmean_weights_list: list, reduce_non_diagonal_ratio: float = 1.0):
            """
            merge parameters of different models with computed regmean weights
            :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
            value is a list of the corresponding parameters of all the models that need to be merged
            :param models_to_merge_regmean_weights_list: list, list of dictionaries with length len(models_to_merge),
            each dictionary records the regmean weights (matrix) of parameters for each model that needs to be merged, key is module name
            :param reduce_non_diagonal_ratio: float, reduce non-diagonal elements in regmean weights by multiplying this scalar
            :return:
            """
            # dict, dictionary of model parameters
            merged_params = {}

            for param_name, param_value_list in models_to_merge_param_dict.items():
                merged_by_regmean = False
                # only perform regmean merging on the "weight" parameter of Linear module
                if param_name.endswith(".weight"):
                    module_name = param_name[:-len(".weight")]
                    if module_name in models_to_merge_regmean_weights_list[0].keys():
                        # two lists with length num_models_to_merge
                        param_multiplied_results, module_regmean_weights_list = [], []
                        for model_idx, model_to_merge_regmean_weights in enumerate(models_to_merge_regmean_weights_list):
                            # Tensor, shape (hidden_dim, hidden_dim)
                            module_regmean_weights = model_to_merge_regmean_weights[module_name]

                            # reduce non-diagonal elements
                            module_regmean_weights = reduce_non_diagonal_elements(regmean_weights=module_regmean_weights, reduce_non_diagonal_ratio=reduce_non_diagonal_ratio)
                            module_regmean_weights_list.append(module_regmean_weights)

                            model_to_merge_param = param_value_list[model_idx]
                            # since the weight shape of Linear module is (output_size, input_size), we need to transpose it
                            param_multiplied_results.append(torch.matmul(module_regmean_weights, model_to_merge_param.transpose(0, 1)))

                        # sum up module_regmean_weights and param_multiplied_results over all individual models
                        sum_module_regmean_weights = sum(module_regmean_weights_list)
                        sum_param_multiplied_results = sum(param_multiplied_results)

                        # get the inverse matrix
                        inv_sum_module_regmean_weights = torch.inverse(sum_module_regmean_weights)
                        # merge parameters with regmean
                        merged_param = torch.matmul(inv_sum_module_regmean_weights, sum_param_multiplied_results)
                        # transpose to the original shape of "weight" in Linear module
                        merged_params[param_name] = merged_param.transpose(0, 1)
                        merged_by_regmean = True
                # use average merging for parameters whose names are not end with ".weight" or not in Linear module
                if not merged_by_regmean:
                    merged_params[param_name] = torch.stack(param_value_list, dim=0).mean(dim=0)

            return merged_params

        # dictionary of list, where key is the parameter name,
        # value is a list of the corresponding parameters of all the models that need to be merged
        models_to_merge_param_dict = defaultdict(list)

        # list of dictionaries with length len(models_to_merge),
        # each dictionary records the regmean weights (matrix) of parameters for each model that needs to be merged
        models_to_merge_regmean_weights_list = []

        # iterate each individual model that needs to be merged
        with torch.no_grad():
            for model_idx, (model_to_merge, trainer, num_regmean_examples) in enumerate(zip(models_to_merge, trainers, nums_regmean_examples)):
                param_dict = {param_name: param_value for param_name, param_value in model_to_merge.named_parameters()}
                # exclude parameter whose name matches element in exclude_param_names_regex
                param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)

                for param_name in param_names_to_merge:
                    models_to_merge_param_dict[param_name].append(param_dict[param_name])

                linear_modules_to_merge = get_modules_to_merge(model=model_to_merge, include_module_types=[nn.Linear])
                handles = []
                # dictionary, regmean matrices for each linear module inputs
                regmean_weights = {}
                # dictionary, number of examples (multiplied the sequence length) used for computing regmean matrices
                num_computed_examples = {}
                # dictionary, number of actual examples used for computing regmean matrices
                num_actual_examples = {}

                for module_name, linear_module_to_merge in linear_modules_to_merge.items():
                    # register a hook in the forward process
                    handle = linear_module_to_merge.register_forward_hook(compute_regmean_weights(module_name=module_name))
                    handles.append(handle)

                train_dataloader = trainer.get_train_dataloader()
                if num_regmean_examples % trainer._train_batch_size != 0:
                    print(f"warning: the number of examples for computing regmean cannot be fully divided by the batch size for model {model_idx}, "
                          "which may lead to a slightly different number of the actually used examples.")
                for step, inputs in tqdm(enumerate(train_dataloader), desc=f"computing regmean weights for model {model_idx}"):
                    if len(num_actual_examples) > 0 and list(num_actual_examples.values())[0] >= num_regmean_examples:
                        break
                    inputs = trainer._prepare_inputs(inputs)
                    outputs = model_to_merge(**inputs)

                models_to_merge_regmean_weights_list.append(regmean_weights)

                # remove the added hook
                for handle in handles:
                    handle.remove()
            # merging with regmean weights
            merged_params = merging_with_regmean_weights(models_to_merge_param_dict=models_to_merge_param_dict, models_to_merge_regmean_weights_list=models_to_merge_regmean_weights_list,
                                                         reduce_non_diagonal_ratio=reduce_non_diagonal_ratio)

        return merged_params

    def ties_merging(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0):
        """
        ties merging method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """
        def task_vector_param_dict_to_single_vector(task_vector: TaskVector):
            """
            convert parameter dictionary in task vector to a single vector
            :param task_vector: TaskVector, task vector
            :return:
            """
            task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
            sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

            # Tensor, shape (num_total_params, )
            return nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()])

        def single_vector_to_task_vector_param_dict(single_vector: torch.Tensor, task_vector: TaskVector):
            """
            convert a single vector to parameter dictionary in task vector
            :param single_vector: Tensor, single vector that contain all parameters in task_vector.task_vector_param_dict
            :param task_vector: TaskVector, task vector
            :return:
            """
            task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
            sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

            nn.utils.vector_to_parameters(single_vector, sorted_task_vector_param_dict.values())

            return sorted_task_vector_param_dict

        def mask_smallest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor, param_value_mask_rate: float = 0.8):
            """
            mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
            :return:
            """
            # num_models_to_merge, num_total_params = flattened_models_to_merge_param.shape
            num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)

            # Tensor, shape (num_models_to_merge, 1), find the num_mask_params-th smallest magnitude element of all the parameters in each individual model
            kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
            # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
            mask = flattened_models_to_merge_param.abs() >= kth_values

            return flattened_models_to_merge_param * mask

        def get_param_signs(flattened_models_to_merge_param: torch.Tensor):
            """
            get the signs for each parameter in flattened_models_to_merge_param, computed over individual models that need to be merged
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :return:
            """
            # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
            param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
            # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
            majority_sign = torch.sign(param_signs.sum(dim=0))
            param_signs[param_signs == 0] = majority_sign
            return param_signs

        def disjoint_merge(flattened_models_to_merge_param: torch.Tensor, param_signs: torch.Tensor):
            """
            disjoint merge that only keeps the parameter values in individual models whose signs are the same as the param_signs, and calculates the averaged parameters.
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :param param_signs: Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
            :return:
            """
            # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
            param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | ((param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
            # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
            param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask

            # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
            num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
            # Tensor, shape (num_total_params, ), the averaged flattened parameters
            merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved, min=1.0)

            return merged_flattened_param

        assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

        flattened_models_to_merge_param = [task_vector_param_dict_to_single_vector(task_vector=task_vector) for task_vector in models_to_merge_task_vectors]
        # Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        flattened_models_to_merge_param = torch.vstack(flattened_models_to_merge_param)

        with torch.no_grad():
            # Tensor, shape (num_models_to_merge, num_total_params), mask the smallest-magnitude parameter values using param_value_mask_rate
            flattened_models_to_merge_param = mask_smallest_magnitude_param_values(flattened_models_to_merge_param=flattened_models_to_merge_param, param_value_mask_rate=param_value_mask_rate)

            # Tensor, shape (num_total_params, ), get the signs for each parameter in flattened_models_to_merge_param
            param_signs = get_param_signs(flattened_models_to_merge_param=flattened_models_to_merge_param)

            # Tensor, shape (num_total_params, ), disjoint merge
            merged_flattened_param = disjoint_merge(flattened_models_to_merge_param=flattened_models_to_merge_param, param_signs=param_signs)

            # merged parameter dictionary
            merged_task_vector_param_dict = single_vector_to_task_vector_param_dict(single_vector=merged_flattened_param, task_vector=models_to_merge_task_vectors[0])
            merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_param_dict)
            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

        return merged_params
    
    # def compute_and_sum_svd_mem_reduction(task_vectors, config):
    def compute_and_sum_svd_mem_reduction(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0):
        """
        Computes the Singular Value Decomposition (SVD) for each vector in the task_vectors,
        reduces the dimensionality of the vectors based on the sv_reduction factor, and concatenate
        the low-rank matrices. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
        Computation of the SVD is performed also for the second operation.

        Args:
            task_vectors (list): A list of task vector objects, where each object contains a
                                dictionary of vectors.
            config (object): Configuration object containing the following attributes:
                            - DATASETS (list): List of datasets.
                            - device (torch.device): The device to perform computations on.

        Returns:
            dict: A dictionary containing the new vectors after SVD computation and merging.
        """
        sv_reduction = 1 / len(models_to_merge)
        device = merged_model.device
        task_vectors = [
            TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge
        ]
        print("Computing SVD...")
        with torch.no_grad():
            new_vector = {}
            for key in task_vectors[0].task_vector_param_dict:
                new_vector[key] = {}
                for i, task_vector in enumerate(task_vectors):
                    vec = task_vector.task_vector_param_dict[key].to(device)

                    if (
                        len(task_vector.task_vector_param_dict[key].shape) == 2
                        and "text_projection" not in key
                    ):
                        u, s, v = torch.linalg.svd(vec, full_matrices=False)

                        if i == 0:
                            print(f"Computed SVD for {key}...")
                            sum_u = torch.zeros_like(u, device=device)
                            sum_s = torch.zeros_like(s, device=device)
                            sum_v = torch.zeros_like(v, device=device)
                        reduced_index_s = int(s.shape[0] * sv_reduction)

                        # select only the first reduced_index_s columns of u and place them
                        sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                            :, :reduced_index_s
                        ]
                        sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                            :reduced_index_s
                        ]
                        # select only the first reduced_index_s rows of v and place them
                        sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                            :reduced_index_s, :
                        ]

                    else:
                        if i == 0:
                            new_vector[key] = vec.clone()
                        else:
                            new_vector[key] += (vec - new_vector[key]) / (i + 1)

                if len(task_vector.task_vector_param_dict[key].shape) == 2 and "text_projection" not in key:
                    u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                    u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                    new_vector[key] = torch.linalg.multi_dot(
                        (
                            u_u,
                            v_u,
                            torch.diag(sum_s),
                            u_v,
                            v_v,
                        )
                    )

            merged_params = {}
            for name, param in merged_model.named_parameters():
                if name in new_vector:
                    merged_params[name] = param + scaling_coefficient * new_vector[name]
                else:
                    merged_params[name] = param
        return merged_params

    def sce(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, select_topk: float = 1.0, scaling_coefficient: float = 0.1):
        """
        SCE (Select, Calculate, Erase) merging method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param select_topk: float, density parameter for selecting top-k variance positions (default: 1.0 means no selection)
        :param scaling_coefficient: float, scaling coefficient (currently not used in SCE, kept for consistency)
        :return:
        """
        task_vectors = [
            TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge
        ]

        merged_params = {}
        for name, param in merged_model.named_parameters():
            # Skip parameters that should be excluded
            if name not in task_vectors[0].task_vector_param_dict:
                continue
                
            deltas_list = [tv.task_vector_param_dict[name] for tv in task_vectors]
            deltas = torch.stack(deltas_list, dim=0)

            # select & calculate
            weights = get_sce_weight(deltas, select_topk)
            weights = torch.tensor(
                weights, dtype=deltas.dtype, device=deltas.device
            )
            while len(deltas.shape) > len(weights.shape):
                weights.unsqueeze_(-1)

            # erase
            mask_dtype = param.dtype
            erase_mask = get_erase_mask(
                deltas,
                mask_dtype=mask_dtype,
            )
            erased_weights = weights * erase_mask
            mixed_delta = (deltas * erased_weights).sum(dim=0)

            # normalize
            divisor = (erased_weights).sum(dim=0)
            divisor[divisor == 0] = 1
            mixed_delta /= divisor

            merged_params[name] = (param.data + mixed_delta).to(param.dtype)

        return merged_params


    def merging_models(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, trainers: list = None, scaling_coefficient: float = 1.0,
                       nums_fisher_examples: list = None, fisher_scaling_coefficients: list = None, normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6,
                       nums_regmean_examples: list = None, reduce_non_diagonal_ratio: float = 1.0, param_value_mask_rate: float = 0.8, select_topk: float = 1.0,
                       weight_format: str = "delta_weight", weight_mask_rates: list = None, use_weight_rescale: bool = True, mask_strategy: str = "random",
                       apply_weight_mask: bool = False, models_use_deepcopy: bool = False):
        """
        model merging methods
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param trainers: list, trainers of individual models
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :param nums_fisher_examples: list, numbers of examples to compute fisher weights
        :param fisher_scaling_coefficients: list, scaling coefficients to merge fisher weights
        :param normalize_fisher_weight: boolean, whether to normalize fisher weights (L2 norm) or not
        :param minimal_fisher_weight: float, the minimal value in fisher weights, used for tackling the potential numerical issues
        :param nums_regmean_examples: list, numbers of examples to compute regmean weights
        :param reduce_non_diagonal_ratio: float, reduce non-diagonal elements in regmean weights by multiplying this scalar
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :param select_topk: float, density parameter for SCE merging (default: 1.0 means no selection)
        :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
        :param weight_mask_rates: list, list of weight mask rates
        :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
        :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
        :param mask_apply_method: str, merging method that the mask strategy applies
        :param models_use_deepcopy: boolean, whether to deepcopy the models
        :return:
        """
        if apply_weight_mask:
            with torch.no_grad():
                if models_use_deepcopy:
                    new_models_to_merge = copy.deepcopy(models_to_merge)
                else:
                    new_models_to_merge = models_to_merge
                for new_model_to_merge, weight_mask_rate in zip(new_models_to_merge, weight_mask_rates):
                    # for each individual model, mask its weight
                    masked_param_dict = mask_model_weights(finetuned_model=new_model_to_merge, pretrained_model=merged_model,
                                                           exclude_param_names_regex=exclude_param_names_regex, weight_format=weight_format,
                                                           weight_mask_rate=weight_mask_rate, use_weight_rescale=use_weight_rescale, mask_strategy=mask_strategy)
                    self.copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)
        if self.merging_method_name == "average_merging":
            merged_params = self.average_merging(models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex)
        elif self.merging_method_name == "task_arithmetic":
            merged_params = self.task_arithmetic(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                                 scaling_coefficient=scaling_coefficient)
        elif self.merging_method_name == "fisher_merging":
            merged_params = self.fisher_merging(models_to_merge=models_to_merge, trainers=trainers, exclude_param_names_regex=exclude_param_names_regex,
                                                nums_fisher_examples=nums_fisher_examples, fisher_scaling_coefficients=fisher_scaling_coefficients,
                                                normalize_fisher_weight=normalize_fisher_weight, minimal_fisher_weight=minimal_fisher_weight)
        elif self.merging_method_name == "regmean_merging":
            merged_params = self.regmean_merging(models_to_merge=models_to_merge, trainers=trainers, exclude_param_names_regex=exclude_param_names_regex,
                                                 nums_regmean_examples=nums_regmean_examples, reduce_non_diagonal_ratio=reduce_non_diagonal_ratio)
        elif self.merging_method_name == "ties_merging":
            merged_params = self.ties_merging(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                              param_value_mask_rate=param_value_mask_rate, scaling_coefficient=scaling_coefficient)
        elif self.merging_method_name == "tsv_merging":
            merged_params = self.compute_and_sum_svd_mem_reduction(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                                                   param_value_mask_rate=param_value_mask_rate, scaling_coefficient=scaling_coefficient)
        elif self.merging_method_name == "sce_merging":
            merged_params = self.sce(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                     select_topk=select_topk, scaling_coefficient=scaling_coefficient)
        else:
            raise NotImplementedError(f"unsupported for merging_method_name {self.merging_method_name}!")
        return merged_params

    def get_merged_model(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, trainers: list = None, scaling_coefficient: float = 1.0,
                         nums_fisher_examples: list = None, fisher_scaling_coefficients: list = None, normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6,
                         nums_regmean_examples: list = None, reduce_non_diagonal_ratio: float = 1.0, param_value_mask_rate: float = 0.8, select_topk: float = 1.0,
                         weight_format: str = "delta_weight", weight_mask_rates: list = None, use_weight_rescale: bool = True, mask_strategy: str = "random",
                         apply_weight_mask: bool = False, models_use_deepcopy: bool = False):
        """
        merge the parameters of models_to_merge to merged_model
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param trainers: list, trainers of individual models
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :param nums_fisher_examples: list, numbers of examples to compute fisher weights
        :param fisher_scaling_coefficients: list, scaling coefficients to merge fisher weights
        :param normalize_fisher_weight: boolean, whether to normalize fisher weights (L2 norm) or not
        :param minimal_fisher_weight: float, the minimal value in fisher weights, used for tackling the potential numerical issues
        :param nums_regmean_examples: list, numbers of examples to compute regmean weights
        :param reduce_non_diagonal_ratio: float, reduce non-diagonal elements in regmean weights by multiplying this scalar
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :param select_topk: float, density parameter for SCE merging (default: 1.0 means no selection)
        :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
        :param weight_mask_rates: list, list of weight mask rates
        :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
        :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
        :param mask_apply_method: str, merging method that the mask strategy applies
        :param models_use_deepcopy: boolean, whether to deepcopy the models
        :return:
        """
        # merged_params, dict of parameters
        merged_params = self.merging_models(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex, trainers=trainers,
                                            nums_fisher_examples=nums_fisher_examples, scaling_coefficient=scaling_coefficient, fisher_scaling_coefficients=fisher_scaling_coefficients,
                                            normalize_fisher_weight=normalize_fisher_weight, minimal_fisher_weight=minimal_fisher_weight,
                                            nums_regmean_examples=nums_regmean_examples, reduce_non_diagonal_ratio=reduce_non_diagonal_ratio, param_value_mask_rate=param_value_mask_rate,
                                            select_topk=select_topk, weight_format=weight_format, weight_mask_rates=weight_mask_rates, use_weight_rescale=use_weight_rescale, mask_strategy=mask_strategy,
                                            apply_weight_mask=apply_weight_mask, models_use_deepcopy=models_use_deepcopy)
        self.copy_params_to_model(params=merged_params, model=merged_model)

        return merged_model


def get_erase_mask(
        delta: torch.Tensor,
        mask_dtype = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.
    """
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    sign_weight = delta.sum(dim=0)
    majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
    del sign_weight

    return sign == majority_sign

def get_sce_mask(deltas, density):
    variance = torch.var(deltas, dim=0, unbiased=False, keepdim=False)
    non_zero_positions_count = torch.count_nonzero(variance)
    k = int(density * non_zero_positions_count)
    mask = torch.zeros_like(variance)
    if k == 0:
        return mask
    assert k > 0, "not gonna zero out the whole tensor buddy"
    variance_abs = variance.abs().view(-1)
    topk_vari = torch.topk(variance_abs, k=k, largest=True)
    mask.view(-1)[topk_vari.indices] = 1
    return mask


def get_sce_weight(deltas, density):
    # select
    if density < 1:
        mask = get_sce_mask(deltas, density)
        deltas = [delta * mask for delta in deltas]
    # calculate
    weights = [torch.sum(delta ** 2).item() / delta.numel() for delta in deltas]
    sum_weights = sum(weights)
    if sum_weights == 0:
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [item / sum_weights for item in weights]
    return weights