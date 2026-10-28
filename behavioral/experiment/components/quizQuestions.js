export const QUIZ_QUESTIONS = [
  {
    id: 'pg1',
    questions: [
      {
        id: 'avgPoints',
        question: 'What is the range of the button values?',
        multiSelect: false,
        answers: ['-1000 to 1000', '0 to 5', '0 to 100'],
        correctAnswer: ['0 to 100'],
      },
      {
        id: 'noise',
        question: 'Does each button return the same value every time or a random number around its value?',
        multiSelect: false,
        answers: ['same number', 'random number around its value'],
        correctAnswer: ['random number around its value'],
      },
    ],
  },
  {
    id: 'pg2',
    questions: [
      {
        id: 'numTrials',
        question: 'How many trials are there per game?',
        multiSelect: false,
        answers: ['10', '20', '50'],
        correctAnswer: ['20'],
      },
      {
        id: 'goal',
        question: 'What is your goal in this game?',
        multiSelect: false,
        answers: ['get to 0 points', 'maximize your points', 'play as fast as possible'],
        correctAnswer: ['maximize your points'],
      },
    ],
  },
]
