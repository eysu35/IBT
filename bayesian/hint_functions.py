import numpy as np


#### Helpers ####
def _type_mask(t_ids, *type_indices):
    mask = np.zeros(len(t_ids), dtype=bool)
    for idx in type_indices:
        mask |= t_ids == idx
    return mask


def _log_type(t_ids, *types):
    return np.where(_type_mask(t_ids, *types), 0.0, -np.inf)


def _best_arm_type(t_ids, rs, *types):
    condition = np.min(
        [
            np.argmax(rs, axis=0) == (rs.shape[0] - 1),
            _type_mask(t_ids, *types),
        ],
        axis=0,
    )
    return np.where(condition, 0.0, -np.inf)


def _param_check(params, key, check_fn):
    return np.array([check_fn(p.get(key)) if key in p else False for p in params])


def _history_stats(kwargs, n_arms):
    actions = kwargs.get("actions")
    if actions is None or len(actions) == 0:
        return None
    actions = np.asarray(actions, dtype=int)
    t = len(actions)
    counts = np.bincount(actions, minlength=n_arms)
    last_action = int(actions[-1])
    run_len = 1
    for i in range(t - 2, -1, -1):
        if actions[i] == last_action:
            run_len += 1
        else:
            break
    since_last = np.full(n_arms, t)
    for i, a in enumerate(actions):
        since_last[a] = t - 1 - i
    return {
        "n_tried": np.count_nonzero(counts),
        "counts": counts,
        "last_action": last_action,
        "run_len": run_len,
        "since_last": since_last,
        "t": t,
    }


def _mean_rewards(kwargs, h):
    obs = kwargs.get("obs")
    if obs is not None and len(obs) > 0:
        return np.nan_to_num(np.nanmean(np.asarray(obs, dtype=float), axis=0), nan=0.0)
    return h["counts"] / h["counts"].sum()


def _switch_policy(t_ids, rs, params=None, **kwargs):
    n_arms, n_samples = rs.shape
    h = _history_stats(kwargs, n_arms)
    if h is None:
        return _log_type(t_ids, _PATCH)
    best = np.argmax(rs, axis=0)
    staleness = h["since_last"] / max(h["t"], 1)  # 1 = long rested, 0 = just chosen
    weights = staleness[best] - 1.0
    return np.where(_type_mask(t_ids, _PATCH), weights, -np.inf)


# Type indices (match TYPE_ORDER in constants.py)
_STAT, _DRIFT, _DELAY, _PATCH = 0, 1, 2, 3
_TWO_CTX, _THREE_CTX, _BERN, _LINCON = 4, 5, 6, 7


#### Stationary ####


def max2_is_best(t_ids, rs, params=None, **kwargs):
    return np.where(np.argmax(rs, axis=0) == (rs.shape[0] - 1), 0.0, -np.inf)


def min2_is_worst(t_ids, rs, params=None, **kwargs):
    return np.where(np.argmin(rs, axis=0) == 0, 0.0, -np.inf)


def explore_then_stick(t_ids, rs, params=None, **kwargs):
    n_arms, n_samples = rs.shape
    h = _history_stats(kwargs, n_arms)
    if h is None:
        return _log_type(t_ids, _STAT, _BERN)
    best = np.argmax(rs, axis=0)
    if h["n_tried"] < n_arms:
        # explore
        preferred = np.isin(best, np.where(h["counts"] == 0)[0])
    else:
        # stick:
        preferred = best == int(np.argmax(_mean_rewards(kwargs, h)))
    weights = np.where(preferred, 0.0, -1.0)
    return np.where(_type_mask(t_ids, _STAT, _BERN), weights, -np.inf)


def max1_better_than_min2_min1_mid(t_ids, rs, params=None, **kwargs):
    condition = np.min([rs[3] > rs[0], rs[3] > rs[1], rs[3] > rs[2]], axis=0)
    return np.where(condition, 0.0, -np.inf)


def one_button_is_best(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _STAT)


#### Drifting ####


def max2_begins_best_but_values_change(t_ids, rs, params=None, **kwargs):
    return _best_arm_type(t_ids, rs, _DRIFT, _PATCH)


def values_drift_small_each_trial(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _DRIFT)


def continue_trying_others_to_detect_changes(t_ids, rs, params=None, **kwargs):
    n_arms, n_samples = rs.shape
    h = _history_stats(kwargs, n_arms)
    if h is None:
        return _log_type(t_ids, _DRIFT)
    staleness = h["since_last"] / max(h["t"], 1)  # 1 = unchecked, 0 = just sampled
    weights = staleness[np.argmax(rs, axis=0)] - 1.0
    return np.where(_type_mask(t_ids, _DRIFT), weights, -np.inf)


def min2_and_min1_begin_worst(t_ids, rs, params=None, **kwargs):
    worst_two = np.argsort(rs, axis=0)[:2, :]
    both_worst = ((worst_two[0] == 0) | (worst_two[1] == 0)) & (
        (worst_two[0] == 1) | (worst_two[1] == 1)
    )
    condition = np.min([both_worst, _type_mask(t_ids, _DRIFT, _PATCH)], axis=0)
    return np.where(condition, 0.0, -np.inf)


def values_change_gradually(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _DRIFT, _PATCH)


#### Context ####


def values_change_for_square_and_circle(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _TWO_CTX)


def explore_buttons_best_for_each_shape(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _TWO_CTX, _THREE_CTX)


def square_max2_best_circle_min2_best(t_ids, rs, params=None, **kwargs):
    best = np.argmax(rs, axis=0)
    n = rs.shape[0]
    ctx = kwargs.get("context")
    if ctx == 0:
        arm_mask = best == n - 1
    elif ctx == 1:
        arm_mask = best == 0
    else:
        arm_mask = (best == 0) | (best == n - 1)
    condition = np.min([arm_mask, _type_mask(t_ids, _TWO_CTX, _THREE_CTX)], axis=0)
    return np.where(condition, 0.0, -np.inf)


def pay_attention_to_shapes(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _TWO_CTX, _THREE_CTX)


def each_shape_different_values(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _TWO_CTX, _THREE_CTX)


def square_G_circle_A_triangle_explore(t_ids, rs, params=None, **kwargs):
    best = np.argmax(rs, axis=0)
    n = rs.shape[0]
    ctx = kwargs.get("context")
    if ctx == 0:  # square: sorted ascending, arm n-1 is best
        arm_mask = best == n - 1
    elif ctx == 1:  # circle: sorted descending, arm 0 is best
        arm_mask = best == 0
    else:  # explore
        arm_mask = np.ones(len(best), dtype=bool)
    condition = np.min([arm_mask, _type_mask(t_ids, _THREE_CTX)], axis=0)
    return np.where(condition, 0.0, -np.inf)


def values_change_under_each_shape(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _TWO_CTX, _THREE_CTX)


def square_circle_sorted_different_directions(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _THREE_CTX)


#### Time Delayed ####


def max2_best_even_if_not_at_first(t_ids, rs, params=None, **kwargs):
    return _best_arm_type(t_ids, rs, _DELAY)


def choose_button_few_trials_in_row(t_ids, rs, params=None, **kwargs):
    n_arms, n_samples = rs.shape
    h = _history_stats(kwargs, n_arms)
    if h is None:
        return _log_type(t_ids, _DELAY)
    best = np.argmax(rs, axis=0)
    weights = np.where(best == h["last_action"], 0.0, -0.25 / (1 + h["run_len"]))
    return np.where(_type_mask(t_ids, _DELAY), weights, -np.inf)


def points_correspond_to_3_trials_ago(t_ids, rs, params=None, **kwargs):
    if params is not None:
        condition = np.min(
            [
                _type_mask(t_ids, _DELAY),
                _param_check(params, "delay_k", lambda k: k == 3),
            ],
            axis=0,
        )
        return np.where(condition, 0.0, -np.inf)
    return _log_type(t_ids, _DELAY)


def time_delay_between_choice_and_points(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _DELAY)


#### Patch Foraging ####


def max2_best_but_decreases(t_ids, rs, params=None, **kwargs):
    return _switch_policy(t_ids, rs, params, **kwargs)


def values_decrease_if_keep_choosing(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _PATCH)


def alternate_good_buttons(t_ids, rs, params=None, **kwargs):
    return _switch_policy(t_ids, rs, params, **kwargs)


def first_pick_most_representative(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _PATCH)


def dont_stick_too_long(t_ids, rs, params=None, **kwargs):
    return _switch_policy(t_ids, rs, params, **kwargs)


#### Bernoulli ####


def max2_most_likely_100_points(t_ids, rs, params=None, **kwargs):
    return _best_arm_type(t_ids, rs, _BERN)


def explore_stick_most_often_returned(t_ids, rs, params=None, **kwargs):
    return explore_then_stick(t_ids, rs, params, **kwargs)


def probability_fixed_for_each_button(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _BERN)


def chance_of_0_even_if_good(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _BERN)


def choose_multiple_times_learn_probability(t_ids, rs, params=None, **kwargs):
    # prescriptive "stay and sample" hurt a normative agent -> just the Bernoulli structure
    return _log_type(t_ids, _BERN)


#### Linear Contextual ####


def third_number_1_most_points(t_ids, rs, params=None, **kwargs):
    if params is not None:
        condition = np.min(
            [
                _type_mask(t_ids, _LINCON),
                _param_check(
                    params, "theta", lambda t: t is not None and np.argmax(t) == 2
                ),
            ],
            axis=0,
        )
        return np.where(condition, 0.0, -np.inf)
    return _log_type(t_ids, _LINCON)


def one_number_leads_to_more_points(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _LINCON)


def three_numbers_combine_to_total(t_ids, rs, params=None, **kwargs):
    return _log_type(t_ids, _LINCON)


def choose_button_third_number_1(t_ids, rs, params=None, **kwargs):
    return third_number_1_most_points(t_ids, rs, params)


def first_number_negative_impact(t_ids, rs, params=None, **kwargs):
    if params is not None:
        condition = np.min(
            [
                _type_mask(t_ids, _LINCON),
                _param_check(params, "theta", lambda t: t is not None and t[0] < 0),
            ],
            axis=0,
        )
        return np.where(condition, 0.0, -np.inf)
    return _log_type(t_ids, _LINCON)


_ALL_FUNCTIONS = [
    max2_is_best,
    min2_is_worst,
    explore_then_stick,
    max1_better_than_min2_min1_mid,
    one_button_is_best,
    max2_begins_best_but_values_change,
    values_drift_small_each_trial,
    continue_trying_others_to_detect_changes,
    min2_and_min1_begin_worst,
    values_change_gradually,
    values_change_for_square_and_circle,
    explore_buttons_best_for_each_shape,
    square_max2_best_circle_min2_best,
    pay_attention_to_shapes,
    each_shape_different_values,
    square_G_circle_A_triangle_explore,
    values_change_under_each_shape,
    square_circle_sorted_different_directions,
    max2_best_even_if_not_at_first,
    choose_button_few_trials_in_row,
    points_correspond_to_3_trials_ago,
    time_delay_between_choice_and_points,
    max2_best_but_decreases,
    values_decrease_if_keep_choosing,
    alternate_good_buttons,
    first_pick_most_representative,
    dont_stick_too_long,
    max2_most_likely_100_points,
    explore_stick_most_often_returned,
    probability_fixed_for_each_button,
    chance_of_0_even_if_good,
    choose_multiple_times_learn_probability,
    third_number_1_most_points,
    one_number_leads_to_more_points,
    three_numbers_combine_to_total,
    choose_button_third_number_1,
    first_number_negative_impact,
]
for _f in _ALL_FUNCTIONS:
    _f.name = _f.__name__.replace("_", " ")

HINT_TEMPLATE_TO_FUNCTION = {
    "stationary": {
        "{max2} is the best.": max2_is_best,
        "{min2} is the worst.": min2_is_worst,
        "Explore all buttons, then stick with the one that gave the most points.": explore_then_stick,
        "{max1} is better than {min2}, {min1}, and {mid}.": max1_better_than_min2_min1_mid,
        "One button is the best.": one_button_is_best,
    },
    "drifting": {
        "{max2} begins as the best, but the button values change over trials.": max2_begins_best_but_values_change,
        "The button values drift by a small number at each trial.": values_drift_small_each_trial,
        "Even after finding a good button, continue trying the others to detect changes.": continue_trying_others_to_detect_changes,
        "{min2} and {min1} begin as the worst buttons.": min2_and_min1_begin_worst,
        "The button values change gradually.": values_change_gradually,
    },
    "two_context": {
        "The button values change for square and circle.": values_change_for_square_and_circle,
        "Explore the buttons to find the best one for each shape.": explore_buttons_best_for_each_shape,
        "When you see a square, {max2} is the best. When you see a circle, {min2} is the best.": square_max2_best_circle_min2_best,
        "Pay attention to the shapes.": pay_attention_to_shapes,
        "Each shape corresponds to a different set of button values.": each_shape_different_values,
    },
    "three_context": {
        "When you see a square, choose G. When you see a circle, choose A. When you see a triangle, explore all buttons to find the best.": square_G_circle_A_triangle_explore,
        "The button values change under each shape.": values_change_under_each_shape,
        "When you see a square or circle, the buttons are sorted in different directions.": square_circle_sorted_different_directions,
        "Explore the buttons to find the best one for each shape.": explore_buttons_best_for_each_shape,
        "Pay attention to the shapes.": pay_attention_to_shapes,
    },
    "time_delayed": {
        "{max2} is the best, even if it doesn't seem like it at first.": max2_best_even_if_not_at_first,
        "Choose a button for a few trials in a row to observe its true value.": choose_button_few_trials_in_row,
        "The points correspond to the choice you made 3 trials ago.": points_correspond_to_3_trials_ago,
        "There is a time delay between your choice and the points you receive.": time_delay_between_choice_and_points,
        "{max2} is the best.": max2_is_best,
    },
    "patch_foraging": {
        "{max2} begins as the best, but the points decrease over time.": max2_best_but_decreases,
        "Each button value will decrease if you keep choosing it, but it will increase over time if not.": values_decrease_if_keep_choosing,
        "Find a few good buttons and alternate between them to maximize your points.": alternate_good_buttons,
        "The points you get on the first pick of a button are the most representative of the button value.": first_pick_most_representative,
        "Do not stick with any one button for too long.": dont_stick_too_long,
    },
    "bernoulli": {
        "{max2} is most likely to return 100 points.": max2_most_likely_100_points,
        "Explore all buttons, then stick to the one that most often returned points.": explore_stick_most_often_returned,
        "The probability for returning points is fixed for each button.": probability_fixed_for_each_button,
        "There is a chance of returning 0 points even if the button is good.": chance_of_0_even_if_good,
        "Choose a button multiple times to learn its probability of returning points.": choose_multiple_times_learn_probability,
    },
    "linear_contextual": {
        "If the third number under the button is 1, you will get the most points.": third_number_1_most_points,
        "One of the three numbers under the buttons always leads to more points.": one_number_leads_to_more_points,
        "The three numbers under each button combine to calculate the total points.": three_numbers_combine_to_total,
        "Choose the button with a 1 as its third number.": choose_button_third_number_1,
        "The first number has a negative impact on the points. Only choose buttons with a 0 as the first number.": first_number_negative_impact,
    },
}
