/**
 * Reconstruct all-5-arm features for linear_contextual trials.
 *
 * The SMILE experiment saves only the *chosen* arm's features per trial.
 * This script replays the seeded PRNG (seedrandom) to regenerate all 5
 * arms' feature vectors, verified against the saved chosen-arm features.
 *
 * Usage:  node reconstruct_armfts.js [input.json]
 * Output: reconstruct_armfts_out.json (array of {user_id, game_idx, trial, allArmFts})
 */

const fs = require('fs');
const path = require('path');
const seedrandom = require(path.join(__dirname, '..', 'node_modules', 'seedrandom', 'seedrandom.js'));

const NUM_TRIALS = 20;
const NUM_ARMS = 5;
const NUM_BANDIT_TYPES = 8;

function generateArmFeatures() {
    const features = [];
    const bestArmIndex = Math.floor(Math.random() * NUM_ARMS);
    for (let i = 0; i < NUM_ARMS; i++) {
        const f1 = Math.random() < 0.5 ? 0 : 1;
        const f2 = Math.random() < 0.5 ? 0 : 1;
        const f3 = i === bestArmIndex ? 1 : 0;
        features.push([f1, f2, f3]);
    }
    return features;
}

function reconstructGame(seedID, routeIndex) {
    const seed = `${seedID}-instructed_bandits/${routeIndex}`;
    seedrandom(seed, { global: true });
    // First call: game type selection
    const typeIndex = Math.floor(Math.random() * NUM_BANDIT_TYPES);
    if (typeIndex !== 7) return null; // not linearContextual
    // Generate features
    const armFts = [];
    for (let t = 0; t < NUM_TRIALS; t++) {
        armFts.push(generateArmFeatures());
    }
    return armFts;
}

// --- main ---
const inputFile = process.argv[2] || path.join(__dirname,
    'eysu35-instructed_rl-freeze-completed-2026-02-10T05-48-04.json');
const raw = JSON.parse(fs.readFileSync(inputFile, 'utf8'));
const users = Array.isArray(raw) ? raw : [raw];

const output = [];
let verified = 0, mismatches = 0;

for (let uid = 0; uid < users.length; uid++) {
    const user = users[uid];
    const seedID = user.seedID;
    if (!seedID) continue;

    const study = user.studyData || [];
    // Find game boundaries (id === 0 marks trial 0 of each game)
    const trialEntries = study.filter(e => e && typeof e === 'object' && e.type);

    let gameIdx = -1;
    let trialInGame = 0;
    for (let i = 0; i < trialEntries.length; i++) {
        const e = trialEntries[i];
        if (e.id === 0) {
            gameIdx++;
            trialInGame = 0;
        }
        if (e.type !== 'linearContextual') {
            trialInGame++;
            continue;
        }

        // On first trial of this game, reconstruct all features
        if (trialInGame === 0) {
            const allFts = reconstructGame(seedID, gameIdx);
            if (!allFts) {
                console.error(`PRNG type mismatch: user ${uid}, game ${gameIdx}`);
                mismatches++;
                trialInGame++;
                continue;
            }
            // Verify and emit
            for (let t = 0; t < NUM_TRIALS && (i + t) < trialEntries.length; t++) {
                const trial = trialEntries[i + t];
                if (trial.type !== 'linearContextual') break;
                const savedFts = trial.armFts;
                const choice = trial.choice;
                // Verify: saved features should match the chosen arm's features
                if (savedFts && choice != null) {
                    const reconstructed = allFts[t][choice];
                    const match = JSON.stringify(reconstructed) === JSON.stringify(savedFts);
                    if (match) verified++;
                    else {
                        console.error(`Mismatch: user ${uid}, game ${gameIdx}, trial ${t}: ` +
                            `saved=${JSON.stringify(savedFts)} vs reconstructed=${JSON.stringify(reconstructed)}`);
                        mismatches++;
                    }
                }
                output.push({
                    user_id: uid,
                    game_idx: gameIdx,
                    trial: t,
                    allArmFts: allFts[t]
                });
            }
        }
        trialInGame++;
    }
}

const outPath = path.join(__dirname, 'reconstruct_armfts_out.json');
fs.writeFileSync(outPath, JSON.stringify(output));
console.log(`Wrote ${output.length} trial records to ${outPath}`);
console.log(`Verified: ${verified}, Mismatches: ${mismatches}`);
