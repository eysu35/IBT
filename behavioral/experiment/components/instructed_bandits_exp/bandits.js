import { add, get } from 'lodash'
import { getGaussian, processHint } from './utils.js'

export const rewardFunctions = {
  stationary: getRewardStationary,
  drifting: getRewardDrifting,
  time_delayed: getRewardTimeDelayed,
  patch_foraging: getRewardPatchForaging,
  bernoulli: getRewardBernoulli,
  twoContext: getRewardTwoContext,
  threeContext: getRewardThreeContext,
  linearContextual: getRewardLinearContextual,
}
export const BANDIT_TYPES = [
  getStationary,
  getDrifting,
  getTimeDelayed,
  getPatchForaging,
  getBernoulli,
  getTwoContext,
  getThreeContext,
  getLinearContextual,
]

/* shared variables */
const NUM_TRIALS = 20

/* stationary mab */
function getStationary() {
  const trialData = {
    arms: [
      [10, 20, 40, 60, 80],
      [80, 10, 20, 40, 60],
      [60, 80, 10, 20, 40],
      [40, 60, 80, 10, 20],
      [20, 40, 60, 80, 10],
    ],
    hints: [
      'No hint.',
      '{max2} is the best.',
      '{min2} is the worst.',
      'Explore all buttons, then stick with the one that gave the most points.',
      '{max1} is better than {min2}, {min1}, and {mid}.',
      'One button is the best.',
    ],
  }

  const arms = trialData.arms[Math.floor(Math.random() * trialData.arms.length)]
  const hint = trialData.hints[Math.floor(Math.random() * trialData.hints.length)]

  return { type: 'stationary', arms: arms, ogHint: hint, hint: processHint(hint, arms), nTrials: NUM_TRIALS }
}

export function getRewardStationary(arms, choices, _) {
  const mu = arms[choices[choices.length - 1]]
  const z = getGaussian()
  return [z * 5 + mu, arms]
}

/* drifting mab */
function getDrifting() {
  const trialData = {
    arms: [
      [10, 20, 40, 60, 80],
      [80, 10, 20, 40, 60],
      [60, 80, 10, 20, 40],
      [40, 60, 80, 10, 20],
      [20, 40, 60, 80, 10],
    ],
    hints: [
      'No hint.',
      '{max2} begins as the best, but the button values change over trials.',
      'The button values drift by a small number at each trial.',
      'Even after finding a good button, continue trying the others to detect changes.',
      '{min2} and {min1} begin as the worst buttons.',
      'The button values change gradually.',
    ],
  }

  const arms = trialData.arms[Math.floor(Math.random() * trialData.arms.length)]
  const hint = trialData.hints[Math.floor(Math.random() * trialData.hints.length)]

  return { type: 'drifting', arms: arms, ogHint: hint, hint: processHint(hint, arms), nTrials: NUM_TRIALS }
}

export function getRewardDrifting(arms, choices, _) {
  const mu = arms[choices[choices.length - 1]]
  const z = getGaussian()
  const new_arms = arms.map((mean) => Math.round((mean + getGaussian() * 5) * 100) / 100)
  return [z * 5 + mu, new_arms]
}

/* time delayed mab */
function getTimeDelayed() {
  const trialData = {
    arms: [
      [10, 20, 40, 60, 80],
      [80, 10, 20, 40, 60],
      [60, 80, 10, 20, 40],
      [40, 60, 80, 10, 20],
      [20, 40, 60, 80, 10],
    ],
    hints: [
      'No hint.',
      '{max2} is the best.',
      "{max2} is the best, even if it doesn't seem like it at first.",
      'Choose a button for a few trials in a row to observe its true value.',
      'The points correspond to the choice you made 3 trials ago.',
      'There is a time delay between your choice and the points you receive.',
    ],
  }

  const arms = trialData.arms[Math.floor(Math.random() * trialData.arms.length)]
  const hint = trialData.hints[Math.floor(Math.random() * trialData.hints.length)]

  return { type: 'time_delayed', arms: arms, ogHint: hint, hint: processHint(hint, arms), nTrials: NUM_TRIALS }
}

export function getRewardTimeDelayed(arms, choices, _, t = 3) {
  if (choices.length < t) {
    return [0.0, arms]
  }

  const mu = arms[choices[choices.length - t]]
  const z = getGaussian()
  return [z * 5 + mu, arms]
}

/* patch foraging mab */
function getPatchForaging() {
  const trialData = {
    arms: [
      [10, 20, 40, 60, 80],
      [80, 10, 20, 40, 60],
      [60, 80, 10, 20, 40],
      [40, 60, 80, 10, 20],
      [20, 40, 60, 80, 10],
    ],
    hints: [
      'No hint.',
      '{max2} begins as the best, but the points decrease over time.',
      'Each button value will decrease if you keep choosing it, but it will increase over time if not.',
      'Find a few best buttons and alternate between them to maximize your points.',
      'The points you get on the first pick of a button is the most representative of the button value.',
      "Don't stick with one button for too long.",
    ],
  }

  const arms = trialData.arms[Math.floor(Math.random() * trialData.arms.length)]
  const hint = trialData.hints[Math.floor(Math.random() * trialData.hints.length)]

  return { type: 'patch_foraging', arms: arms, ogHint: hint, hint: processHint(hint, arms), nTrials: NUM_TRIALS }
}

export function getRewardPatchForaging(arms, choices, gtArms, changeRate = 0.2) {
  const mu = arms[choices[choices.length - 1]]
  const z = getGaussian()
  const reward = z * 5 + mu

  const new_arms = [...arms]
  for (let i = 0; i < arms.length; i++) {
    if (i === choices[choices.length - 1]) {
      const depleted = arms[i] * Math.exp(-changeRate)
      new_arms[i] = Math.max(0, depleted)
    } else {
      const replenished = arms[i] * Math.exp(changeRate)
      new_arms[i] = Math.min(replenished, gtArms[i])
    }
  }
  return [reward, new_arms]
}

/* bernoulli mab */
function getBernoulli() {
  const trialData = {
    arms: [
      [10, 20, 40, 60, 80],
      [80, 10, 20, 40, 60],
      [60, 80, 10, 20, 40],
      [40, 60, 80, 10, 20],
      [20, 40, 60, 80, 10],
    ],
    hints: [
      'No hint.',
      '{max2} is most likely to return 100 points.',
      'Explore all buttons, then stick to the ones that more often returned points.',
      'The probability for returning points is fixed for each button.',
      'There is a chance of returning 0 points even if the button is good.',
      'Choose a button multiple times to learn its probability of returning points.',
    ],
  }

  const arms = trialData.arms[Math.floor(Math.random() * trialData.arms.length)]
  const hint = trialData.hints[Math.floor(Math.random() * trialData.hints.length)]

  return {
    type: 'bernoulli',
    arms: arms,
    ogHint: hint,
    hint: processHint(hint, arms),
    nTrials: NUM_TRIALS * 2,
  }
}

export function getRewardBernoulli(arms, choices, _) {
  const p = arms[choices[choices.length - 1]] / 100.0
  const win = Math.random() < p ? 1 : 0
  return [win * 100, arms]
}

/* two-context mab */
function getTwoContext() {
  let currContext = 0
  function gen_contexts() {
    const p_switch = 0.25
    if (Math.random() < p_switch) {
      currContext = 1 - currContext
    }
    return currContext
  }

  const trialData = {
    arms: [
      [
        [10, 20, 40, 60, 80],
        [80, 60, 40, 20, 10],
      ],
      [
        [80, 10, 20, 40, 60],
        [10, 80, 60, 40, 20],
      ],
      [
        [60, 80, 10, 20, 40],
        [20, 10, 80, 60, 40],
      ],
      [
        [40, 60, 80, 10, 20],
        [40, 20, 10, 80, 60],
      ],
      [
        [20, 40, 60, 80, 10],
        [60, 40, 20, 10, 80],
      ],
    ],
    hints: [
      'No hint.',
      'The button values change for square and circle.',
      'Explore the buttons under both shapes to find the best one in each.',
      'When you see a square, {max2} is the best. When you see a circle, {min2} is the best.',
      'Pay attention to the shapes.',
      'Each shape corresponds to a different set of button values.',
    ],
  }

  const arms = trialData.arms[Math.floor(Math.random() * trialData.arms.length)]
  const hint = trialData.hints[Math.floor(Math.random() * trialData.hints.length)]
  const contexts = Array.from({ length: NUM_TRIALS * 2 }, () => gen_contexts())
  const contextArms = contexts.map((c) => arms[c])

  return {
    type: 'twoContext',
    arms: contextArms,
    contexts: contexts,
    ogHint: hint,
    hint: processHint(hint, arms[0]),
    nTrials: NUM_TRIALS * 2,
  }
}

export function getRewardTwoContext(arms, choices, _) {
  const mu = arms[choices[choices.length - 1]]
  const z = getGaussian()
  return [z * 5 + mu, arms]
}

/* three-context mab */
function getThreeContext() {
  let currContext = Math.floor(Math.random() * 3)
  function gen_contexts() {
    const p_switch = 0.25
    if (Math.random() < p_switch) {
      const options = [0, 1, 2].filter((c) => c !== currContext)
      currContext = options[Math.floor(Math.random() * options.length)]
    }

    return currContext
  }
  const trialData = {
    arms: [
      [
        [10, 20, 40, 60, 80],
        [80, 60, 40, 20, 10],
        [10, 60, 20, 40, 80],
      ],
      [
        [10, 20, 40, 60, 80],
        [80, 60, 40, 20, 10],
        [20, 10, 80, 40, 60],
      ],
      [
        [10, 20, 40, 60, 80],
        [80, 60, 40, 20, 10],
        [60, 40, 80, 10, 20],
      ],
      [
        [10, 20, 40, 60, 80],
        [80, 60, 40, 20, 10],
        [80, 20, 10, 60, 40],
      ],
      [
        [10, 20, 40, 60, 80],
        [80, 60, 40, 20, 10],
        [40, 80, 20, 60, 10],
      ],
    ],
    hints: [
      'No hint.',
      'When you see a square, choose G. When you see a circle, choose A. When you see a triangle, explore all buttons to find the best.',
      'Explore the buttons in all shapes to find the best one in each.',
      'The button values change under each shape.',
      'When you see a square or circle, the buttons are sorted in different directions.',
      'Pay attention to the shapes.',
    ],
  }

  const arms = trialData.arms[Math.floor(Math.random() * trialData.arms.length)]
  const hint = trialData.hints[Math.floor(Math.random() * trialData.hints.length)]
  const contexts = Array.from({ length: NUM_TRIALS * 2 }, () => gen_contexts())
  const contextArms = contexts.map((c) => arms[c])

  return {
    type: 'threeContext',
    arms: contextArms,
    contexts: contexts,
    ogHint: hint,
    hint: hint,
    nTrials: NUM_TRIALS * 2,
  }
}

export function getRewardThreeContext(arms, choices, _) {
  const mu = arms[choices[choices.length - 1]]
  const z = getGaussian()
  return [z * 5 + mu, arms]
}

/* linear contextual mab */
function getLinearContextual() {
  function generateArmFeatures() {
    const features = []
    const numArms = 5
    const bestArmIndex = Math.floor(Math.random() * numArms)

    for (let i = 0; i < numArms; i++) {
      const f1 = Math.random() < 0.5 ? 0 : 1
      const f2 = Math.random() < 0.5 ? 0 : 1
      const f3 = i === bestArmIndex ? 1 : 0

      features.push([f1, f2, f3])
    }
    return features
  }

  const trialData = {
    hints: [
      'No hint.',
      'If the third number under the button is 1, you will get the most points.',
      'One of the three numbers under the buttons always leads to more points.',
      'The three numbers under each button combine to calculate the total points.',
      'Choose the button with a 1 as its third number.',
      'The first number has a negative impact on the points. Only choose buttons with a 0 as the first number.',
    ],
  }

  const theta = [-2, 0, 5] // hard coded thetas
  const armFts = Array.from({ length: NUM_TRIALS }, () => generateArmFeatures())
  const hint = trialData.hints[Math.floor(Math.random() * trialData.hints.length)]

  return {
    type: 'linearContextual',
    theta: theta,
    armFts: armFts,
    ogHint: hint,
    hint: hint,
    nTrials: NUM_TRIALS,
  }
}

export function getRewardLinearContextual(arm_fts, choices, theta) {
  const fts = arm_fts[choices[choices.length - 1]]
  return fts.reduce((sum, f, i) => sum + f * theta[i], 0)
}
