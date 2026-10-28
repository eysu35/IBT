from __future__ import annotations
import math
from typing import Any


def normalize_type(t: Any) -> str | None | float:
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return t
    s = str(t).strip()
    s = s.replace("linearContextual", "linear_contextual")
    s = s.replace("timeDelayed", "time_delayed")
    s = s.replace("patchForaging", "patch_foraging")
    s = s.replace("twoContext", "two_context")
    s = s.replace("threeContext", "three_context")
    return s


TYPE_ORDER = [
    "stationary",
    "drifting",
    "time_delayed",
    "patch_foraging",
    "two_context",
    "three_context",
    "bernoulli",
    "linear_contextual",
]

IDEAL_PRIORS = {
    "arm_prior_loc": 45.0,  # ≈ mean of (10, 20, 40, 60, 80)
    "arm_prior_scale": 20.0,  # covers true arm-mean range
    "drift_eps": 2.0,  # true experiment drift_epsilon
    "eps_stdev": 1.0,  # tight around truth
    "lam": 0.2,  # true experiment change_rate
    "lam_stdev": 0.08,  # tight around truth
}


_HINT_ORDER_WITH_LABELS: list[tuple[str, list[str]]] = [
    # Stationary
    ("{max2} is the best.", ["V"]),
    ("{min2} is the worst.", ["V"]),
    ("Explore all buttons, then stick with the one that gave the most points.", ["P"]),
    ("{max1} is better than {min2}, {min1}, and {mid}.", ["V"]),
    ("One button is the best.", ["S"]),
    # Drifting
    ("{max2} begins as the best, but the button values change over trials.", ["V"]),
    ("The button values drift by a small number at each trial.", ["S"]),
    (
        "Even after finding a good button, continue trying the others to detect changes.",
        ["P"],
    ),
    ("{min2} and {min1} begin as the worst buttons.", ["V"]),
    ("The button values change gradually.", ["S"]),
    # Two Context
    ("The button values change for square and circle.", ["S"]),
    ("Explore the buttons to find the best one for each shape.", ["P"]),
    (
        "When you see a square, {max2} is the best. When you see a circle, {min2} is the best.",
        ["V"],
    ),
    ("Pay attention to the shapes.", ["P"]),
    ("Each shape corresponds to a different set of button values.", ["S"]),
    # Three Context
    (
        "When you see a square, choose G. When you see a circle, choose A. When you see a triangle, explore all buttons to find the best.",
        ["V"],
    ),
    ("The button values change under each shape.", ["S"]),
    (
        "When you see a square or circle, the buttons are sorted in different directions.",
        ["S"],
    ),
    (
        "Explore the buttons to find the best one for each shape.",
        ["P"],
    ),  # shared with two_context
    ("Pay attention to the shapes.", ["P"]),  # shared with two_context
    # Time Delayed
    ("{max2} is the best, even if it doesn't seem like it at first.", ["V"]),
    ("Choose a button for a few trials in a row to observe its true value.", ["P"]),
    ("The points correspond to the choice you made 3 trials ago.", ["S"]),
    ("There is a time delay between your choice and the points you receive.", ["S"]),
    ("{max2} is the best.", ["V"]),  # shared with stationary
    # Patch Foraging
    ("{max2} begins as the best, but the points decrease over time.", ["V"]),
    (
        "Each button value will decrease if you keep choosing it, but it will increase over time if not.",
        ["S"],
    ),
    (
        "Find a few good buttons and alternate between them to maximize your points.",
        ["P"],
    ),
    (
        "The points you get on the first pick of a button are the most representative of the button value.",
        ["S"],
    ),
    ("Do not stick with any one button for too long.", ["P"]),
    # Bernoulli
    ("{max2} is most likely to return 100 points.", ["V"]),
    (
        "Explore all buttons, then stick to the one that most often returned points.",
        ["P"],
    ),
    ("The probability for returning points is fixed for each button.", ["S"]),
    ("There is a chance of returning 0 points even if the button is good.", ["S"]),
    (
        "Choose a button multiple times to learn its probability of returning points.",
        ["P"],
    ),
    # Linear Contextual
    ("If the third number under the button is 1, you will get the most points.", ["V"]),
    ("One of the three numbers under the buttons always leads to more points.", ["S"]),
    (
        "The three numbers under each button combine to calculate the total points.",
        ["S"],
    ),
    ("Choose the button with a 1 as its third number.", ["P"]),
    (
        "The first number has a negative impact on the points. Only choose buttons with a 0 as the first number.",
        ["S"],
    ),
]

HINT_LABELS: dict[str, list[str]] = {"No hint.": []}
for h, L in _HINT_ORDER_WITH_LABELS:
    if h not in HINT_LABELS:
        HINT_LABELS[h] = L


MODEL_COLORS = {
    "human": "#1f77b4",
    "bayesian": "#9467bd",
    "centaur": "#ff7f0e",
    "centaur_IBT": "#d62728",
    "llama": "#2ca02c",
}
