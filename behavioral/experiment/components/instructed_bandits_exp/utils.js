/* gaussian sampling function */

export function getGaussian() {
  const u = 1 - Math.random()
  const v = Math.random()
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
  return z
}

/* hint processing function */
export function processHint(hint, arms) {
  const armsWithIndices = arms.map((value, index) => ({ value, index }))
  armsWithIndices.sort((a, b) => a.value - b.value)
  const positionalMapping = {
    min2: armsWithIndices[0].index, // Second smallest
    min1: armsWithIndices[1].index, // Smallest value
    mid: armsWithIndices[2].index, // Middle value
    max1: armsWithIndices[3].index, // Largest value
    max2: armsWithIndices[4].index, // Second largest
  }

  console.log(positionalMapping)

  const buttonMapping = {
    0: 'A',
    1: 'S',
    2: 'D',
    3: 'F',
    4: 'G',
  }

  const processed = hint.replace(/{(\w+)}/g, (fullMatch, placeholder) => {
    const idx = positionalMapping[placeholder]
    const button = buttonMapping[idx]
    return button !== undefined ? button : fullMatch
  })

  return processed
}

/* gen two context arms */
export function getContextTwo(arms) {
  // take the set of arms and reorder them so the min is in place of the max, max is in place of the min, and so forth
}
