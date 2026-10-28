<script setup>
import { ref, computed } from 'vue'
import useViewAPI from '@/core/composables/useViewAPI'
import { Button } from '@/uikit/components/ui/button'
import { rewardFunctions, BANDIT_TYPES } from './bandits.js'

const api = useViewAPI()

const props = defineProps({
  gameNumber: {
    type: Number,
    required: true,
  },
  totalGames: {
    type: Number,
    required: true,
  },
})

// /* Define global variables */
const chist = ref([])
const rhist = ref([])
const contexthist = ref([])
const hasHint = ref(false)
const hintUnderstanding = ref(5)
const sliderTouched = ref(false)
const NUM_COLS = 3
const NUM_GAMES_PER_PAGE = 1
const trials = api.steps
  .append(
    Array.from({ length: NUM_GAMES_PER_PAGE }, (_, i) => ({
      id: `game_${i}`,
    }))
  )
  .forEach((row, index) => {
    const sampleBandit = BANDIT_TYPES[Math.floor(Math.random() * BANDIT_TYPES.length)]
    const banditConfig = sampleBandit()
    // Check if hint for this bandit
    if (banditConfig.hint && banditConfig.hint !== 'No hint.') {
      hasHint.value = true
    }

    if (banditConfig.type === 'twoContext' || banditConfig.type === 'threeContext') {
      for (let i = 0; i < banditConfig.nTrials; i++) {
        row.append({
          id: i,
          type: banditConfig.type,
          ogArms: banditConfig.arms[0],
          arms: banditConfig.arms[i],
          contexts: banditConfig.contexts[i],
          ogHint: banditConfig.ogHint,
          hint: banditConfig.hint,
          choice: null,
          reward: null,
          rt: null,
          nTrials: banditConfig.nTrials,
        })
      }
    } else if (banditConfig.type === 'linearContextual') {
      for (let i = 0; i < banditConfig.nTrials; i++) {
        row.append({
          id: i,
          type: banditConfig.type,
          theta: banditConfig.theta,
          armFts: banditConfig.armFts[i],
          ogHint: banditConfig.ogHint,
          hint: banditConfig.hint,
          choice: null,
          reward: null,
          rt: null,
          nTrials: banditConfig.nTrials,
        })
      }
    } else {
      for (let i = 0; i < banditConfig.nTrials; i++) {
        row.append({
          id: i,
          type: banditConfig.type,
          ogArms: banditConfig.arms,
          arms: banditConfig.arms,
          ogHint: banditConfig.ogHint,
          hint: banditConfig.hint,
          choice: null,
          reward: null,
          rt: null,
          nTrials: banditConfig.nTrials,
        })
      }
    }
  })
  .append({ id: 'nextGame' })

if (props.gameNumber === props.totalGames - 1) {
  trials.append({ id: 'summary' })
}

if (!api.isTimerStarted()) {
  api.startTimer()
}

/* compute total reward */
const totalReward = computed(() => {
  return rhist.value.reduce((acc, reward) => acc + reward, 0)
})

/* make columns for table visualization */
const historyColumns = computed(() => {
  if (chist.value.length === 0) {
    return []
  }

  const maxTrials = api.stepData.nTrials
  if (!maxTrials) return [[], [], []]

  const rowsPerColumn = Math.ceil(maxTrials / NUM_COLS)

  let fullHistory
  if (
    api.stepData.type === 'twoContext' ||
    api.stepData.type === 'threeContext' ||
    api.stepData.type === 'linearContextual'
  ) {
    fullHistory = chist.value.map((choice, index) => ({
      trial: index + 1,
      context: contexthist.value[index],
      button: buttonLabels[choice],
      reward: rhist.value[index],
    }))
  } else {
    fullHistory = chist.value.map((choice, index) => ({
      trial: index + 1,
      button: buttonLabels[choice],
      reward: rhist.value[index],
    }))
  }
  const col1 = fullHistory.slice(0, rowsPerColumn)
  const col2 = fullHistory.slice(rowsPerColumn, rowsPerColumn * 2)
  const col3 = fullHistory.slice(rowsPerColumn * 2, rowsPerColumn * 3)
  return [col1, col2, col3]
})

/* see if need to collect hint understanding data */
// const hasHint = computed(() => {
//   const hint = api.stepData.hint
//   return hint && hint !== 'No hint.'
// })

/* did user touch slider */
function handleSliderInput() {
  sliderTouched.value = true
}

/* Process one user choice by selecting the correct reward function
and sending as input the choice and arm means */
function processChoice(choiceIndex) {
  if (api.blockIndex == 0) {
    chist.value = []
    rhist.value = []
    contexthist.value = []
  }
  const step = api.stepData
  chist.value.push(choiceIndex)
  api.stepData.choice = choiceIndex
  const rewardFunc = rewardFunctions[step.type]

  let reward
  let new_arms
  // Special case bandits
  if (step.type === 'linearContextual') {
    reward = rewardFunc(step.armFts, chist.value, step.theta)
    api.stepData.armFts = step.armFts[choiceIndex]
    contexthist.value.push(step.armFts[choiceIndex])
  }
  // General case
  else {
    [reward, new_arms] = rewardFunc(step.arms, chist.value, step.ogArms)
  }

  // If need shapes in contexthist
  if (step.type === 'twoContext' || step.type === 'threeContext') {
    contexthist.value.push(contextShape.value)
  }
  api.stepData.reward = reward
  api.stepData.rt = api.elapsedTime()
  rhist.value.push(reward)

  api.recordStep()
  api.goNextStep()

  if (step.type !== 'twoContext' && step.type !== 'threeContext' && step.type !== 'linearContextual') {
    api.stepData.arms = new_arms
  }
}

function finish() {
  // Only record hint understanding if a hint was actually present
  if (hasHint.value) {
    api.stepData.hintUnderstanding = hintUnderstanding.value
  } else {
    api.stepData.hintUnderstanding = null
  }

  api.recordStep()
  api.goNextView()
}

const buttonLabels = ['A', 'S', 'D', 'F', 'G']

/* Compute the context shape (0=Square, 1=Circle, 2=Triangle) */
const contextShape = computed(() => {
  const step = api.stepData
  if (step.type === 'twoContext' || step.type === 'threeContext') {
    const contextValue = parseInt(step.contexts)
    switch (contextValue) {
      case 0:
        return 'Square'
      case 1:
        return 'Circle'
      case 2:
        return 'Triangle'
      default:
        return 'Unknown Shape'
    }
  }
  return 'N/A'
})

/* Compute the arm features */
const armFeatures = computed(() => {
  const step = api.stepData

  if (step && step.type === 'linearContextual') {
    return step.armFts
  }
  return null
})

const stop = api.onKeyDown(
  ['a', 'A', 's', 'S', 'd', 'D', 'f', 'F', 'g', 'G', 'Enter'],
  (e) => {
    if (api.stepData.id !== 'summary') {
      e.preventDefault()
      const keyMap = { a: 0, s: 1, d: 2, f: 3, g: 4 }
      const choice = keyMap[e.key.toLowerCase()]

      if (choice !== undefined) {
        processChoice(choice)
      }
    }
    console.log("HELLOOOOO")
    console.log(hasHint.value)
    // if in between games
    if (api.stepData.id === 'nextGame' && e.key === 'Enter') {
      e.preventDefault()
      if (sliderTouched.value || !hasHint.value) {
        stop()
        finish()
      }
    }

    // if ending the game
    if (api.stepData.id === 'summary') {
      stop()
    }
  },
  { dedupe: true }
)
</script>
<template>
  <!-- display task -->
  <div class="mx-20">
    <div v-if="api.stepData.id !== 'summary' && api.stepData.id !== 'nextGame'" class="text-center">
      <!-- header -->
      <h2 class="text-lg font-bold text-green-600 mb-1">Game {{ props.gameNumber }} Trial {{ api.blockIndex + 1 }}</h2>

      <div v-if="api.stepData.hint !== 'No hint.'"
        class="max-w-sm mx-auto p-2 mb-2 bg-green-100 border border-green-300 text-green-800 rounded-md">
        <h3 class="font-bold text-sm mb-1">Here's a hint!</h3>
        <p class="text-sm leading-snug">{{ api.stepData.hint }}</p>
      </div>

      <!-- display shapes -->
      <div v-if="api.stepData.type === 'twoContext' || api.stepData.type === 'threeContext'" class="mb-2">
        <div class="flex justify-center items-center h-10">
          <div v-if="contextShape === 'Square'" class="w-10 h-10 border-2 border-blue-500 bg-blue-500"></div>

          <div v-else-if="contextShape === 'Circle'"
            class="w-10 h-10 rounded-full border-2 border-blue-500 bg-blue-500"></div>

          <svg v-else-if="contextShape === 'Triangle'" class="w-14 h-14 drop-shadow-sm" viewBox="0 0 24 24" fill="none"
            xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L2 22H22L12 2Z" fill="#3b82f6" />
          </svg>
        </div>
      </div>

      <!-- display arm fts -->
      <div v-if="api.stepData.type === 'linearContextual'" class="flex justify-center space-x-4">
        <div v-for="(label, index) in buttonLabels" :key="index" class="flex flex-col items-center">
          <Button size="lg" variant="outline" @click="processChoice(index)">{{ label }}</Button>
          <div v-if="armFeatures && armFeatures[index]"
            class="mt-2 text-xs font-mono text-gray-700 leading-tight flex space-x-1">
            <span v-for="(feature, fIndex) in armFeatures[index]" :key="fIndex">{{ Math.round(feature) }}</span>
          </div>
        </div>
      </div>

      <!-- display buttons -->
      <div v-else class="flex justify-center space-x-4">
        <Button size="lg" variant="outline" @click="processChoice(0)">A</Button>
        <Button size="lg" variant="outline" @click="processChoice(1)">S</Button>
        <Button size="lg" variant="outline" @click="processChoice(2)">D</Button>
        <Button size="lg" variant="outline" @click="processChoice(3)">F</Button>
        <Button size="lg" variant="outline" @click="processChoice(4)">G</Button>
      </div>

      <p class="text-muted-foreground p-3">(Click a button or press the corresponding key)</p>

      <!-- contextual table -->
      <div v-if="
        api.stepData.type === 'twoContext' ||
        api.stepData.type === 'threeContext' ||
        api.stepData.type === 'linearContextual'
      " class="h-72 rounded-lg">
        <div class="flex space-x-2 justify-center" v-if="historyColumns.length > 0">
          <div class="" v-for="(column, colIndex) in historyColumns" :key="colIndex">
            <table class="text-sm text-left table-auto">
              <thead class="text-xs text-gray-700 uppercase bg-gray-50">
                <tr>
                  <th class="px-1 py-1 font-medium border-b">Trial</th>
                  <th class="px-1 py-1 font-medium border-b text-center">Info</th>
                  <th class="px-1 py-1 font-medium border-b">Button</th>
                  <th class="px-1 py-1 font-medium border-b">Reward</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="item in column" :key="item.trial" class="border-b odd:bg-white even:bg-gray-50">
                  <td class="px-1 py-1">{{ item.trial }}</td>

                  <td class="px-1 py-1">
                    <div class="flex justify-center items-center">
                      <div v-if="item.context === 'Square'" class="w-6 h-6 border-2 border-blue-500 bg-blue-500"></div>

                      <div v-else-if="item.context === 'Circle'"
                        class="w-6 h-6 rounded-full border-2 border-blue-500 bg-blue-500"></div>

                      <svg v-else-if="item.context === 'Triangle'" class="w-6 h-6" viewBox="0 0 24 24" fill="none"
                        xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L2 22H22L12 2Z" fill="#3b82f6" />
                      </svg>

                      <span v-else>{{ item.context }}</span>
                    </div>
                  </td>

                  <td class="px-1 py-1">{{ item.button }}</td>
                  <td class="px-1 py-1">{{ item.reward.toFixed(2) }}</td>
                </tr>
              </tbody>
              <tfoot v-if="colIndex === historyColumns.length - 1">
                <tr class="font-semibold bg-gray-100">
                  <td class="px-1 py-1" colspan="2"></td>
                  <td class="px-1 py-1 text-right">Total</td>
                  <td class="px-1 py-1">{{ totalReward.toFixed(2) }}</td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      </div>

      <!-- normal table -->
      <div v-else class="h-72 rounded-lg">
        <div class="flex space-x-2 justify-center" v-if="historyColumns.length > 0">
          <div class="" v-for="(column, colIndex) in historyColumns" :key="colIndex">
            <table class="text-sm text-left table-auto">
              <thead class="text-xs text-gray-700 uppercase bg-gray-50">
                <tr>
                  <th class="px-1 py-1 font-medium border-b">Trial</th>
                  <th class="px-1 py-1 font-medium border-b">Button</th>
                  <th class="px-1 py-1 font-medium border-b">Reward</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="item in column" :key="item.trial" class="border-b odd:bg-white even:bg-gray-50">
                  <td class="px-1 py-1">{{ item.trial }}</td>
                  <td class="px-1 py-1">{{ item.button }}</td>
                  <td class="px-1 py-1">{{ item.reward.toFixed(2) }}</td>
                </tr>
              </tbody>
              <tfoot v-if="colIndex === historyColumns.length - 1">
                <tr class="font-semibold bg-gray-100">
                  <td class="px-1 py-1"></td>
                  <td class="px-1 py-1 text-right">Total</td>
                  <td class="px-1 py-1">{{ totalReward.toFixed(2) }}</td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- between games -->
    <div v-else-if="api.stepData.id === 'nextGame'" class="text-center">
      <div v-if="hasHint" class="max-w-md mx-auto mb-10 p-6 bg-gray-50 rounded-xl border border-gray-200">
        <label for="hint-slider" class="block mb-4 font-medium text-gray-900">
          How well did you understand the hint in this game?
        </label>

        <input id="hint-slider" type="range" min="0" max="10" v-model="hintUnderstanding" @input="handleSliderInput"
          class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600" />

        <div class="flex justify-between text-xs text-muted-foreground mt-2 px-1">
          <span>Did not understand at all</span>
          <span>Understood perfectly</span>
        </div>

        <p v-if="!sliderTouched" class="mt-4 text-sm text-amber-600 font-medium animate-pulse">
          Please adjust the slider to continue.
        </p>
      </div>

      <div v-else class="mb-10">
        <p class="text-xl font-medium">Game Complete!</p>
      </div>

      <p class="text-lg text-muted-foreground mb-4">Click to play the next game!</p>

      <Button variant="default" size="lg" @click="finish()" :disabled="hasHint && !sliderTouched"
        :class="{ 'opacity-50 cursor-not-allowed': hasHint && !sliderTouched }">
        Continue
        <i-lucide-arrow-right class="w-4 h-4 ml-2" />
      </Button>
    </div>
  </div>
</template>

<style scoped>
.rewardhistory {
  width: 500px;
  margin: auto;
}
</style>
