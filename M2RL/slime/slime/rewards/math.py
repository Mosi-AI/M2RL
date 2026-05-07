def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def math_opensource_reward(pred, label, data_name='aime24'):
    from slime.rewards.math_opensource_utils.parser import parse_ground_truth, STRIP_EXCEPTIONS, extract_answer, strip_string
    from slime.rewards.math_opensource_utils.parser import choice_answer_clean
    from slime.rewards.math_opensource_utils.grader import math_equal_process

    label = str(label)
    prediction = extract_answer(pred, data_name)
    prediction = strip_string(prediction, skip_unit=data_name in STRIP_EXCEPTIONS)

    # cleaning choice results
    if label and prediction not in ["A", "B", "C", "D", "E"]:
            prediction = choice_answer_clean(prediction)
    elif is_multi_choice(label) and not is_multi_choice(prediction):
            # remove any non-choice char
            prediction = "".join(
                [c for c in prediction if c in ["A", "B", "C", "D", "E"]]
            )

    params = [(prediction, label)]
    result = math_equal_process(params[0])
        
    return float(result)