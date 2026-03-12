import argparse
import sys
import os
import shutil
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_merging_methods.merging_methods import MergingMethod
from utils.utils import set_random_seed
from utils.load_config import cache_dir


def get_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger, merging_method: MergingMethod):
    """
    get the performance of merging method named merging_method_name
    :param args: ArgumentParser, input argument parser
    :param finetuned_model_names: list, names of finetuned models
    :param models_to_merge: list, individual models that need to be merged
    :param trainers: list, trainers of individual models
    :param logger: Logger, logger
    :param merging_method: MergingMethod, the mering method
    :param tokenizers: list of tokenizers
    :return:
    """
    logger.info(f"configuration is {args}")

    try:
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name, device_map="cpu")
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name)
    except:
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name, cache_dir=cache_dir, device_map="cpu")
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name)

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    merged_model = pretrained_model
    merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[],
                                                   trainers=trainers,
                                                   scaling_coefficient=args.scaling_coefficient,
                                                   nums_fisher_examples=None,
                                                   fisher_scaling_coefficients=None,
                                                   normalize_fisher_weight=None,
                                                   minimal_fisher_weight=None,
                                                   nums_regmean_examples=None,
                                                   reduce_non_diagonal_ratio=None,
                                                #    param_value_mask_rate=None,
                                                   weight_format=args.weight_format,
                                                   weight_mask_rates=args.weight_mask_rates,
                                                   use_weight_rescale=args.use_weight_rescale,
                                                   mask_strategy=args.mask_strategy,
                                                   apply_weight_mask=args.apply_weight_mask,
                                                   models_use_deepcopy=False)

    # since the tokenizers of different tasks are different, we need to save them (together with the model) separately
    logger.info(f"saving models at {args.save_model_path}...")
    merged_model.save_pretrained(save_directory=args.save_model_path)
    pretrained_tokenizer.save_pretrained(save_directory=args.save_model_path)
    logger.info(f"models are saved at {args.save_model_path}")
    del merged_model, pretrained_model, pretrained_tokenizer


parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument("--pretrained_model_name", type=str, help="name of the pretrained model")
parser.add_argument("--models_to_merge", type=str, help="names of models to merge")
parser.add_argument("--merging_method_name", type=str, default="average_merging", help="name of the method to merge models", choices=["average_merging", "ties_merging", "tsv_merging", "sce_merging", "task_arithmetic"])
parser.add_argument("--scaling_coefficient", type=float, default=1.0, help="scaling coefficient to merge the task vector")
parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])

parser.add_argument("--weight_mask_rate", type=float, default=0.2, help="weight mask rate")
parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
parser.add_argument("--apply_weight_mask", type=bool, default=False, help="merging method that the mask strategy applies")

parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=sys.maxsize)
parser.add_argument("--tensor_parallel_size", type=int, default=1, help="numbers of gpus to use")
args = parser.parse_args()


if __name__ == "__main__":
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    finetuned_model_names = args.models_to_merge.split(",")
    print("models_to_merge: ", finetuned_model_names)
    PendingDeprecationWarning

    args.weight_mask_rates = [args.weight_mask_rate for _ in range(len(finetuned_model_names))]

    if args.merging_method_name in ["average_merging", "sce_merging"]:
        args.save_model_name = f"{args.merging_method_name}"
    elif args.apply_weight_mask:
        args.save_model_name = f"dare_{args.merging_method_name}_scaling_coefficient_{args.scaling_coefficient}"
    else:
        args.save_model_name = f"{args.merging_method_name}_scaling_coefficient_{args.scaling_coefficient}"
    args.save_model_path = os.path.join("./merged_models", args.save_model_name)

    save_merge_log_path = f"./save_merge_llm_logs/{args.save_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")

    models_to_merge = []
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)
    for finetuned_model_name in finetuned_model_names:
        print("loading model: ", finetuned_model_name)
        finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=finetuned_model_name, device_map="cpu")
        models_to_merge.append(finetuned_model)

    get_merge_performance(args=args, models_to_merge=models_to_merge, trainers=[None for _ in range(len(finetuned_model_names))], logger=logger, merging_method=merging_method)
