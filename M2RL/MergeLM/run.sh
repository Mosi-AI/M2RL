# rm -rf added_tokens.json chat_template.jinja special_tokens_map.json
# cp ../temp/* ./

# pip install fraction
# "average_merging", "ties_merging", "tsv_merging", "sce_merging", "task_arithmetic"
MODEL_ROOT=/cpfs01/liziheng/checkpoints
pretrained_model=/cpfs01/haoqingwang/slime_output/Qwen3-4B_SFT_Further/iter_0000999_hf

math_model=/cpfs01/liziheng/checkpoints/Qwen3-4B-math-blend-32k-grpo-lr2e_6/iter_0000399_hf
code_model=/cpfs01/liziheng/checkpoints/Qwen3-4B-code-blend-32k-grpo-pass_all-lr2e_6/iter_0000399_hf
sci_model=/cpfs01/liziheng/checkpoints/Qwen3-4B-science-blend-32k-grpo-lr2e_6/iter_0000199_hf
if_model=/cpfs01/liziheng/checkpoints/Qwen3-4B-if-blend-32k-grpo-lr2e_6/iter_0000399_hf
agent_model=/cpfs01/liziheng/checkpoints/Qwen3-4B-workbench-blend-32k-grpo-lr2e_6/iter_0000399_hf

merge_models=$math_model,$code_model,$sci_model,$if_model,$agent_model
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python merge_llms.py --pretrained_model_name $pretrained_model --models_to_merge $merge_models --merging_method_name average_merging
python merge_llms.py --pretrained_model_name $pretrained_model --models_to_merge $merge_models --merging_method_name ties_merging
python merge_llms.py --pretrained_model_name $pretrained_model --models_to_merge $merge_models --merging_method_name sce_merging
python merge_llms.py --pretrained_model_name $pretrained_model --models_to_merge $merge_models --merging_method_name task_arithmetic
python merge_llms.py --pretrained_model_name $pretrained_model --models_to_merge $merge_models --merging_method_name ties_merging --apply_weight_mask True --use_weight_rescale
python merge_llms.py --pretrained_model_name $pretrained_model --models_to_merge $merge_models --merging_method_name task_arithmetic --apply_weight_mask True --use_weight_rescale
