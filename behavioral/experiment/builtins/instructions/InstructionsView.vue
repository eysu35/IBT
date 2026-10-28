<script setup>
import { computed } from 'vue'
import useViewAPI from '@/core/composables/useViewAPI'
import { Button } from '@/uikit/components/ui/button'
import { ConstrainedPage } from '@/uikit/layouts'

/**
 * Instructions view component for displaying experiment instructions
 * Handles different instruction versions based on experimental conditions
 */
const api = useViewAPI()

/**
 * Get the instruction version condition from the experiment configuration
 * @type {string} The instruction version condition ('1', '2', '3', or undefined)
 */
const cond = api.getConditionByName('instructionsVersion')

/**
 * Computed property that returns the appropriate instruction text based on the condition
 * @returns {string} The instruction text for the current condition
 */
// const instText = computed(() => {
//   if (cond === '1') {
//     return ""
//   }
//   if (cond === '2') {
//     return 'instructions version 2'
//   }
//   if (cond === '3') {
//     return 'instructions version 3'
//   }
//   return 'no condition set'
// })

const instText = `<i>Read carefully then take an understanding quiz.</i> <br><br> You will play a simple game in which you have to make many choices between five buttons. When you choose each button, it will give back some number of points from 0 to 100, but with some randomness around its number. In some games, you will see shapes on the screen or numbers next to the buttons that you should consider when making your choice.<br><br>In each game, you will have 20 or more choices to make to get as many points as you can. You will play 15 rounds of this game, and each game has a different set of button numbers.<br><br>Finally, in some games, you will receive a hint which will help you in the game. In others, you will be given no hint.<br><br>You will also receive a bonus based on the number of points you won on one randomly selected choice from all the games. Each point is $0.01, which means you can win a bonus of up to $1.00.`

/**
 * Navigate to the next view in the experiment flow
 * @param {boolean} [goto] - Optional parameter for navigation control (unused in current implementation)
 */
function finish(goto) {
  api.goNextView()
}
</script>

<template>
  <!-- Main instruction container with responsive layout -->
  <ConstrainedPage :width="800" :height="800" :responsiveUI="true">
    <!-- Instruction content wrapper -->
    <div class="w-full h-full p-8 flex flex-col overflow-y-auto">
      <!-- Instruction header with icon -->
      <h1 class="text-2xl font-bold mb-4">
        <i-material-symbols-integration-instructions class="inline-block mr-2 text-3xl" /> Instructions
      </h1>

      <!-- Dynamic instruction text based on condition -->
      <p class="text-left text-lg mb-4" v-html="instText"></p>

      <!-- Visual separator -->
      <hr class="border-gray-300 my-4" />

      <!-- Navigation button container -->
      <div class="flex justify-end">
        <Button variant="default" @click="finish()">
          next
          <i-fa6-solid-arrow-right />
        </Button>
      </div>
    </div>
  </ConstrainedPage>
</template>
