/* Instructed Bandit Tasks: playable task, result charts, Bayesian demo. No dependencies. */
(() => {
  const TYPES = ['stationary', 'drifting', 'time_delayed', 'patch_foraging', 'two_context', 'three_context', 'bernoulli', 'linear_contextual']
  const LABEL = {
    stationary: 'Stationary', drifting: 'Drifting', time_delayed: 'Time delayed', patch_foraging: 'Patch foraging',
    two_context: 'Two context', three_context: 'Three context', bernoulli: 'Bernoulli', linear_contextual: 'Linear contextual',
  }
  const LETTERS = ['A', 'S', 'D', 'F', 'G']
  const SHAPES = ['square', 'circle', 'triangle']
  const ARMSETS = [[10, 20, 40, 60, 80], [80, 10, 20, 40, 60], [60, 80, 10, 20, 40], [40, 60, 80, 10, 20], [20, 40, 60, 80, 10]]
  const TWO = [
    [[10, 20, 40, 60, 80], [80, 60, 40, 20, 10]], [[80, 10, 20, 40, 60], [10, 80, 60, 40, 20]], [[60, 80, 10, 20, 40], [20, 10, 80, 60, 40]],
    [[40, 60, 80, 10, 20], [40, 20, 10, 80, 60]], [[20, 40, 60, 80, 10], [60, 40, 20, 10, 80]],
  ]
  const THREE = [
    [[10, 20, 40, 60, 80], [80, 60, 40, 20, 10], [10, 60, 20, 40, 80]], [[10, 20, 40, 60, 80], [80, 60, 40, 20, 10], [20, 10, 80, 40, 60]],
    [[10, 20, 40, 60, 80], [80, 60, 40, 20, 10], [60, 40, 80, 10, 20]], [[10, 20, 40, 60, 80], [80, 60, 40, 20, 10], [80, 20, 10, 60, 40]],
    [[10, 20, 40, 60, 80], [80, 60, 40, 20, 10], [40, 80, 20, 60, 10]],
  ]
  const HINTS = {
    stationary: ['{max2} is the best.', '{min2} is the worst.', 'Explore all buttons, then stick with the one that gave the most points.', '{max1} is better than {min2}, {min1}, and {mid}.', 'One button is the best.'],
    drifting: ['{max2} begins as the best, but the button values change over trials.', 'The button values drift by a small number at each trial.', 'Even after finding a good button, continue trying the others to detect changes.', '{min2} and {min1} begin as the worst buttons.', 'The button values change gradually.'],
    time_delayed: ['{max2} is the best.', "{max2} is the best, even if it doesn't seem like it at first.", 'Choose a button for a few trials in a row to observe its true value.', 'The points correspond to the choice you made 3 trials ago.', 'There is a time delay between your choice and the points you receive.'],
    patch_foraging: ['{max2} begins as the best, but the points decrease over time.', 'Each button value will decrease if you keep choosing it, but it will increase over time if not.', 'Find a few good buttons and alternate between them to maximize your points.', 'The points you get on the first pick of a button are the most representative of the button value.', 'Do not stick with any one button for too long.'],
    two_context: ['The button values change for square and circle.', 'Explore the buttons to find the best one for each shape.', 'When you see a square, {max2} is the best. When you see a circle, {min2} is the best.', 'Pay attention to the shapes.', 'Each shape corresponds to a different set of button values.'],
    three_context: ['When you see a square, choose G. When you see a circle, choose A. When you see a triangle, explore all buttons to find the best.', 'Explore the buttons to find the best one for each shape.', 'The button values change under each shape.', 'When you see a square or circle, the buttons are sorted in different directions.', 'Pay attention to the shapes.'],
    bernoulli: ['{max2} is most likely to return 100 points.', 'Explore all buttons, then stick to the one that most often returned points.', 'The probability for returning points is fixed for each button.', 'There is a chance of returning 0 points even if the button is good.', 'Choose a button multiple times to learn its probability of returning points.'],
    linear_contextual: ['If the third number under the button is 1, you will get the most points.', 'One of the three numbers under the buttons always leads to more points.', 'The three numbers under each button combine to calculate the total points.', 'Choose the button with a 1 as its third number.', 'The first number has a negative impact on the points. Only choose buttons with a 0 as the first number.'],
  }
  const AGENTS = [['human', 'People'], ['bayesian', 'Bayesian'], ['centaur', 'Centaur']]
  const CLS = { human: 'human', bayesian: 'bayes', centaur: 'centaur' }
  const DASH = { human: '', bayesian: 'dashed', centaur: 'dotdash' }

  // ---------- helpers
  const $ = (id) => document.getElementById(id)
  const NS = 'http://www.w3.org/2000/svg'
  function svg(tag, attrs = {}, children = []) {
    const e = document.createElementNS(NS, tag)
    for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, v)
    for (const c of children) e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c)
    return e
  }
  const cw = (el, lo = 300, hi = 960) => Math.max(lo, Math.min(hi, Math.floor(el.clientWidth) || hi))
  function h(tag, attrs = {}, children = []) {
    const e = document.createElement(tag)
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') e.className = v
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v)
      else if (v !== null && v !== undefined) e.setAttribute(k, v)
    }
    for (const c of [].concat(children)) e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c)
    return e
  }
  const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random())
  const pick = (a) => a[Math.floor(Math.random() * a.length)]
  const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? '' : (+v).toFixed(d))
  const wrap = (text, width) => {
    const words = text.split(' '), lines = []
    let cur = ''
    for (const w of words) {
      if ((cur + ' ' + w).trim().length > width && cur) { lines.push(cur); cur = w } else cur = (cur + ' ' + w).trim()
    }
    if (cur) lines.push(cur)
    return lines
  }
  function ranks(arms) {
    const order = arms.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value)
    return { min2: order[0].index, min1: order[1].index, mid: order[2].index, max1: order[3].index, max2: order[4].index }
  }
  function processHint(hint, arms) {
    const r = ranks(arms)
    return hint.replace(/{(\w+)}/g, (m, k) => (r[k] === undefined ? m : LETTERS[r[k]]))
  }
  const byHint = (o, hint) => o[hint] || o[numberHint(hint, null)]  // Centaur's logs spell 'choose G' as 'choose 5'
  const tickStep = (lo, hi) => [1, 2, 5, 10, 20, 25, 50, 100].find((s) => (hi - lo) / s <= 8) || 200
  const exampleHint = (hint) => processHint(hint, ARMSETS[0])
  const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)

  // tooltip
  const tip = $('tooltip')
  function bindTips(root) {
    root.addEventListener('mousemove', (e) => {
      const t = e.target.closest('[data-tip]')
      if (!t) { tip.hidden = true; return }
      tip.innerHTML = t.dataset.tip
      tip.hidden = false
      const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8)
      tip.style.left = x + 'px'
      tip.style.top = e.clientY + 14 + 'px'
    })
    root.addEventListener('mouseleave', () => { tip.hidden = true })
  }
  function chips(container, items, selected, onSelect) {
    container.replaceChildren()
    for (const [key, label] of items) {
      const b = h('button', { class: 'chip', role: 'tab', 'aria-selected': String(key === selected), onclick: () => {
        for (const c of container.children) c.setAttribute('aria-selected', String(c.dataset.key === key))
        onSelect(key)
      } }, label)
      b.dataset.key = key
      container.appendChild(b)
    }
  }
  function shapeSVG(ctx, size = 28) {
    const c = 'var(--human)'
    if (ctx === 0) return svg('svg', { viewBox: '0 0 24 24', width: size, height: size }, [svg('rect', { x: 3, y: 3, width: 18, height: 18, fill: c })])
    if (ctx === 1) return svg('svg', { viewBox: '0 0 24 24', width: size, height: size }, [svg('circle', { cx: 12, cy: 12, r: 9.5, fill: c })])
    return svg('svg', { viewBox: '0 0 24 24', width: size, height: size }, [svg('path', { d: 'M12 2.5 2.5 21.5h19Z', fill: c })])
  }
  function axes(g, x, y, xt, yt, W, H, pad, opts = {}) {
    for (const v of yt) {
      g.appendChild(svg('line', { class: 'grid', x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v) }))
      g.appendChild(svg('text', { class: 'muted label', x: pad.l - 8, y: y(v) + 4, 'text-anchor': 'end' }, [String(v)]))
    }
    for (const v of xt) g.appendChild(svg('text', { class: 'muted label', x: x(v), y: H - pad.b + 16, 'text-anchor': 'middle' }, [String(v)]))
    if (opts.xlabel) g.appendChild(svg('text', { class: 'muted label', x: (pad.l + W - pad.r) / 2, y: H - 4, 'text-anchor': 'middle' }, [opts.xlabel]))
    if (opts.ylabel) g.appendChild(svg('text', { class: 'muted label', transform: `translate(12 ${(pad.t + H - pad.b) / 2}) rotate(-90)`, 'text-anchor': 'middle' }, [opts.ylabel]))
  }
  function marker(cls, cx, cy, agent, r = 5) {
    if (agent === 'human') return svg('circle', { class: cls, cx, cy, r })
    if (agent === 'bayesian') return svg('rect', { class: cls + ' open', x: cx - r, y: cy - r, width: 2 * r, height: 2 * r })
    return svg('path', { class: cls + ' open', d: `M${cx} ${cy - r * 1.25} L${cx + r * 1.25} ${cy} L${cx} ${cy + r * 1.25} L${cx - r * 1.25} ${cy} Z` })
  }
  function legend(items) {
    return h('div', { class: 'legend' }, items.map(([cls, label]) => h('span', { class: cls }, [h('i'), label])))
  }

  // ---------- task simulation (ported from the experiment's bandits.js)
  function sampleEnv(type) {
    if (type === 'two_context') return { sets: pick(TWO) }
    if (type === 'three_context') return { sets: pick(THREE) }
    if (type === 'linear_contextual') return {}
    return { arms: pick(ARMSETS) }
  }
  const labelArms = (env) => env.arms || (env.sets && env.sets[0]) || null
  function makeGame(type, hintRaw, env = sampleEnv(type)) {
    const g = { type, hintRaw, choices: [], rewards: [], regrets: [], contextsSeen: [], t: 0, spread: 70 }
    if (type === 'two_context' || type === 'three_context') {
      const sets = env.sets
      const k = sets.length
      let ctx = type === 'two_context' ? 0 : Math.floor(Math.random() * 3)
      g.nTrials = 40
      g.contexts = Array.from({ length: 40 }, () => {
        if (Math.random() < 0.25) ctx = k === 2 ? 1 - ctx : pick([0, 1, 2].filter((c) => c !== ctx))
        return ctx
      })
      g.sets = sets
      g.hint = type === 'two_context' ? processHint(hintRaw, sets[0]) : hintRaw
      g.armsAt = (t) => sets[g.contexts[t]]
    } else if (type === 'linear_contextual') {
      g.nTrials = 20
      g.theta = [-2, 0, 5]
      g.spread = 7
      g.feats = Array.from({ length: 20 }, () => {
        const best = Math.floor(Math.random() * 5)
        return LETTERS.map((_, i) => [Math.random() < 0.5 ? 0 : 1, Math.random() < 0.5 ? 0 : 1, i === best ? 1 : 0])
      })
      g.hint = hintRaw
    } else {
      g.nTrials = type === 'bernoulli' ? 40 : 20
      g.ogArms = env.arms
      g.arms = [...g.ogArms]
      g.hint = processHint(hintRaw, g.ogArms)
    }
    g.step = (c) => {
      const t = g.t
      let reward, regret
      if (type === 'linear_contextual') {
        const f = g.feats[t]
        reward = f[c].reduce((s, v, i) => s + v * g.theta[i], 0)
        regret = 5 - reward
      } else if (type === 'two_context' || type === 'three_context') {
        const arms = g.armsAt(t)
        reward = arms[c] + 5 * gauss()
        regret = Math.max(...arms) - arms[c]
        g.contextsSeen.push(g.contexts[t])
      } else if (type === 'bernoulli') {
        reward = Math.random() < g.arms[c] / 100 ? 100 : 0
        regret = Math.max(...g.arms) - g.arms[c]
      } else if (type === 'time_delayed') {
        const seq = [...g.choices, c]
        reward = seq.length < 3 ? 0 : g.arms[seq[seq.length - 3]] + 5 * gauss()
        regret = Math.max(...g.arms) - g.arms[c]
      } else {
        reward = g.arms[c] + 5 * gauss()
        regret = Math.max(...g.arms) - g.arms[c]
        if (type === 'drifting') g.arms = g.arms.map((m) => Math.round((m + 5 * gauss()) * 100) / 100)
        if (type === 'patch_foraging') g.arms = g.arms.map((m, i) => (i === c ? Math.max(0, m * Math.exp(-0.2)) : Math.min(m * Math.exp(0.2), g.ogArms[i])))
      }
      g.choices.push(c)
      g.rewards.push(reward)
      g.regrets.push((100 * regret) / g.spread)
      g.t += 1
      return reward
    }
    return g
  }

  // ---------- data
  const DATA = {}
  async function load() {
    const names = ['curves', 'hint_effects', 'timing', 'centaur']
    const res = await Promise.all(names.map((n) => fetch(`data/${n}.json`).then((r) => r.json())))
    names.forEach((n, i) => (DATA[n] = res[i]))
  }

  // ---------- play
  let game = null, taskSel = 'stationary', hintSel = 'none', env = sampleEnv('stationary')
  function hintOptions(type, keep = false) {
    const sel = $('hint-select'), arms = labelArms(env)
    sel.replaceChildren(h('option', { value: 'none' }, 'No hint'))
    for (const hint of HINTS[type]) sel.appendChild(h('option', { value: hint }, arms ? processHint(hint, arms) : hint))
    sel.value = keep ? hintSel : 'none'
    hintSel = sel.value
  }
  function renderArms(container, onPick, enabled) {
    container.replaceChildren()
    LETTERS.forEach((L, i) => container.appendChild(h('button', { class: 'arm', disabled: enabled ? null : '', onclick: () => onPick(i) }, L)))
  }
  function renderGame() {
    const started = !!game && game.t < game.nTrials
    $('trial-counter').textContent = game ? `Trial ${Math.min(game.t + 1, game.nTrials)} / ${game.nTrials}` : `${LABEL[taskSel]} · ${taskSel === 'bernoulli' || taskSel === 'two_context' || taskSel === 'three_context' ? 40 : 20} trials`
    $('score').textContent = game ? `${fmt(game.rewards.reduce((a, b) => a + b, 0), 0)} points` : ''
    const hb = $('hint-box')
    hb.hidden = !game || game.hintRaw === 'none'
    if (game && game.hintRaw !== 'none') hb.textContent = game.hint
    const ctx = $('context')
    ctx.hidden = !(started && (game.type === 'two_context' || game.type === 'three_context'))
    if (!ctx.hidden) ctx.replaceChildren(shapeSVG(game.contexts[game.t]))
    const ft = $('features')
    ft.hidden = !(started && game.type === 'linear_contextual')
    if (!ft.hidden) ft.replaceChildren(...game.feats[game.t].map((f) => h('span', {}, f.join(' '))))
    renderArms($('arms'), choose, started)
    const hist = $('history')
    hist.replaceChildren()
    hist.hidden = !game || game.choices.length === 0
    if (game) game.choices.forEach((c, i) => {
      const li = h('li', {}, [h('span', { class: 'muted' }, String(i + 1))])
      if (game.contextsSeen.length) li.appendChild(h('span', { class: 'shape' }, [shapeSVG(game.contextsSeen[i], 14)]))
      li.appendChild(h('b', {}, LETTERS[c]))
      li.appendChild(h('span', {}, fmt(game.rewards[i], game.type === 'linear_contextual' ? 0 : 1)))
      hist.appendChild(li)
    })
    hist.scrollTop = hist.scrollHeight
    $('start-btn').textContent = game && game.t >= game.nTrials ? 'Play again' : started ? 'Restart' : 'Start'
  }
  function choose(i) {
    if (!game || game.t >= game.nTrials) return
    game.step(i)
    renderGame()
    if (game.t >= game.nTrials) showResult()
  }
  function start() {
    if (game) { env = sampleEnv(taskSel); hintOptions(taskSel, true) }
    game = makeGame(taskSel, hintSel, env)
    $('result').hidden = true
    renderGame()
  }
  function pooled(agentCurves, key) {
    if (key !== 'any') return agentCurves[key] || null
    const keys = Object.keys(agentCurves).filter((k) => k !== 'none')
    const n = keys.reduce((s, k) => s + agentCurves[k].n, 0)
    const T = agentCurves[keys[0]].mean.length
    const mean = Array.from({ length: T }, (_, t) => keys.reduce((s, k) => s + agentCurves[k].n * agentCurves[k].mean[t], 0) / n)
    const ci = Array.from({ length: T }, (_, t) => Math.sqrt(keys.reduce((s, k) => s + (agentCurves[k].n * agentCurves[k].ci[t]) ** 2, 0)) / n)
    return { n, mean, ci }
  }
  function showResult() {
    const c = DATA.curves[game.type]
    const you = game.regrets
    const mine = you.reduce((a, b) => a + b, 0) / you.length
    const ppl = byHint(c.human, game.hintRaw) || c.human.none
    const theirs = ppl.mean.reduce((a, b) => a + b, 0) / ppl.mean.length
    $('result-text').textContent = `Your average regret was ${fmt(mine, 0)}%. People in the same condition averaged ${fmt(theirs, 0)}%.`
    const series = [{ key: 'you', cls: 'you', label: 'You', mean: you, ci: null }]
    for (const [a, label] of AGENTS) {
      const s = byHint(c[a], game.hintRaw)
      if (s) series.push({ key: a, cls: CLS[a], label, mean: s.mean, ci: s.ci, dash: DASH[a] })
    }
    $('result').hidden = false
    $('result-chart').replaceChildren(lineChart(series, { W: cw($('result-chart')), ylabel: 'Regret (%)', xlabel: 'Trial' }), legend([['you', 'You'], ...AGENTS.map(([a, l]) => [CLS[a], l])]))
    $('result').scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    centaurSel = { type: game.type, hint: game.hintRaw }
    for (const c of $('centaur-chips').children) c.setAttribute('aria-selected', String(c.dataset.key === game.type))
    centaurHintOptions(game.type, game.hintRaw)
    renderCentaur()
  }
  function curveMax(series) {
    let ymax = 0
    for (const s of series) s.mean.forEach((m, i) => (ymax = Math.max(ymax, m + (s.ci ? s.ci[i] : 0))))
    return Math.max(10, Math.ceil(ymax / 10) * 10)
  }
  function lineChart(series, opts = {}) {
    const W = opts.W || 640, H = opts.H || 260, pad = { l: 48, r: 12, t: opts.title ? 30 : 12, b: 44 }
    const T = Math.max(...series.map((s) => s.mean.length))
    const ymax = opts.ymax || curveMax(series)
    const x = lin(0, T - 1, pad.l, W - pad.r), y = lin(0, ymax, H - pad.b, pad.t)
    const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': opts.ylabel + ' by trial' })
    const yt = ymax <= 40 ? [0, 10, 20, 30, 40].filter((v) => v <= ymax) : [0, 25, 50, 75, 100].filter((v) => v <= ymax)
    const xt = T > 20 ? [1, 10, 20, 30, 40] : [1, 5, 10, 15, 20]
    axes(root, (v) => x(v - 1), y, xt, yt, W, H, pad, opts)
    if (opts.title) root.appendChild(svg('text', { class: 'label', x: 0, y: 12 }, [opts.title]))
    for (const s of series) {
      if (s.ci) {
        const up = s.mean.map((m, i) => `${x(i)},${y(m + s.ci[i])}`), dn = s.mean.map((m, i) => `${x(i)},${y(Math.max(0, m - s.ci[i]))}`).reverse()
        root.appendChild(svg('polygon', { class: 'fill-' + s.cls, points: [...up, ...dn].join(' ') }))
      }
    }
    for (const s of series) {
      root.appendChild(svg('polyline', { class: `line ${s.cls} ${s.dash || ''}`, points: s.mean.map((m, i) => `${x(i)},${y(m)}`).join(' '), 'stroke-width': s.key === 'you' ? 2.5 : 2 }))
    }
    // hover: nearest trial column
    for (let i = 0; i < T; i++) {
      const rows = series.filter((s) => s.mean[i] !== undefined).map((s) => `<b>${s.label}</b> ${fmt(s.mean[i], 0)}%`).join('<br>')
      const hit = svg('rect', { class: 'hit', x: x(i) - (x(1) - x(0)) / 2, y: pad.t, width: x(1) - x(0), height: H - pad.t - pad.b })
      hit.dataset.tip = `Trial ${i + 1}<br>${rows}`
      root.appendChild(hit)
    }
    return root
  }

  // ---------- finding 1: learning curves, no hint vs any hint
  function renderCurves(type) {
    const c = DATA.curves[type]
    const seriesFor = (key) => AGENTS.map(([a, label]) => {
      const s = pooled(c[a], key)
      return s && { key: a, cls: CLS[a], label, mean: s.mean, ci: s.ci, dash: DASH[a] }
    }).filter(Boolean)
    const none = seriesFor('none'), any = seriesFor('any')
    const ymax = Math.max(curveMax(none), curveMax(any))
    const cells = [h('div', { class: 'chart' }), h('div', { class: 'chart' })]
    $('curve-chart').replaceChildren(h('div', { class: 'two' }, cells), legend(AGENTS.map(([a, l]) => [CLS[a], l])))
    ;[[none, 'No hint'], [any, 'With a hint']].forEach(([series, title], i) => cells[i].appendChild(lineChart(series, { W: cw(cells[i]), H: 270, ymax, title, ylabel: 'Regret (%)', xlabel: 'Trial' })))
    cur.curves = type
  }

  // ---------- finding 2: per-hint effects
  function effectChart(type) {
    const d = DATA.hint_effects[type]
    const rows = [...d.hints, { hint: 'No hint (model vs. people)', stars: '', human: d.nohint.human, bayesian: d.nohint.bayesian, centaur: d.nohint.centaur, base: true }]
    const W = Math.max(560, cw($('effect-chart'))), labelW = W >= 760 ? 310 : 230, pad = { l: labelW + 12, r: 40, t: 10, b: 44 }
    // a row is as tall as its wrapped hint needs
    const wrapped = rows.map((r) => wrap(exampleHint(r.hint), W >= 760 ? 40 : 30)), heights = wrapped.map((l) => Math.max(48, l.length * 17 + 14))
    const tops = heights.map((_, i) => pad.t + heights.slice(0, i).reduce((a, b) => a + b, 0))
    const H = pad.t + heights.reduce((a, b) => a + b, 0) + pad.b
    let lo = d.human_band[0], hi = d.human_band[1]
    for (const r of rows) for (const [a] of AGENTS) if (r[a]) { lo = Math.min(lo, r[a][1]); hi = Math.max(hi, r[a][2]) }
    const span = hi - lo
    lo -= 0.06 * span; hi += 0.06 * span
    const x = lin(lo, hi, pad.l, W - pad.r)
    const root = svg('svg', { class: 'wide', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Regret reduction per hint' })
    root.appendChild(svg('rect', { class: 'band', x: x(d.human_band[0]), y: pad.t, width: x(d.human_band[1]) - x(d.human_band[0]), height: H - pad.t - pad.b }))
    const step = span > 40 ? 20 : span > 16 ? 10 : 5
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      root.appendChild(svg('line', { class: v === 0 ? 'zero' : 'grid', x1: x(v), x2: x(v), y1: pad.t, y2: H - pad.b }))
      root.appendChild(svg('text', { class: 'muted label', x: x(v), y: H - pad.b + 16, 'text-anchor': 'middle' }, [String(v)]))
    }
    root.appendChild(svg('text', { class: 'muted label', x: (pad.l + W - pad.r) / 2, y: H - 4, 'text-anchor': 'middle' }, ['Regret reduction (% of arm spread)']))
    rows.forEach((r, i) => {
      const y0 = tops[i], rowH = heights[i], lines = wrapped[i]
      if (r.base) root.appendChild(svg('line', { class: 'grid', x1: pad.l, x2: W - pad.r, y1: y0, y2: y0, 'stroke-dasharray': '2 3' }))
      const t = svg('text', { class: r.base ? 'muted label' : 'label', x: labelW, y: y0 + rowH / 2 - (lines.length - 1) * 8 + 5, 'text-anchor': 'end' })
      lines.forEach((l, j) => t.appendChild(svg('tspan', { x: labelW, dy: j ? 17 : 0 }, [l])))
      root.appendChild(t)
      const offsets = { human: -11, bayesian: 0, centaur: 11 }
      let rowMax = -Infinity
      for (const [a, label] of AGENTS) {
        const v = r[a]
        if (!v) continue
        if (r.base && a === 'human') continue
        const cy = y0 + rowH / 2 + offsets[a]
        rowMax = Math.max(rowMax, v[2])
        root.appendChild(svg('line', { class: `err ${CLS[a]}`, x1: x(v[1]), x2: x(v[2]), y1: cy, y2: cy }))
        root.appendChild(marker(CLS[a], x(v[0]), cy, a, a === 'human' ? 6 : 5))
        const hit = svg('rect', { class: 'hit', x: Math.min(x(v[1]), x(v[0]) - 8), y: cy - 6, width: Math.max(16, x(v[2]) - x(v[1])), height: 12 })
        hit.dataset.tip = `<b>${label}</b> ${fmt(v[0])} [${fmt(v[1])}, ${fmt(v[2])}]`
        root.appendChild(hit)
      }
      if (r.stars) root.appendChild(svg('text', { class: 'star', x: x(rowMax) + 8, y: y0 + rowH / 2 - 7 }, [r.stars]))
    })
    return root
  }
  function renderEffects(type) {
    cur.effects = type
    $('effect-chart').replaceChildren(effectChart(type), legend([...AGENTS.map(([a, l]) => [CLS[a], l]), ['band', 'Grey band: people without a hint, 95% CI']]))
  }

  // ---------- finding 3: hint timing
  function timingPanel(row, band, ylim, W, nLines) {
    const hint = exampleHint(row.hint)
    const pad = { l: 36, r: 8, t: 13 + 17 * nLines, b: 44 }, H = pad.t + 132 + pad.b
    const onsets = [0, 4, 8, 12]
    const x = lin(0, 12, pad.l + 10, W - pad.r - 10), y = lin(ylim[0], ylim[1], H - pad.b, pad.t)
    const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': hint })
    const title = svg('text', { class: 'label', x: 0, y: 12 })
    wrap(hint, Math.floor(W / 7.1)).forEach((l, j) => title.appendChild(svg('tspan', { x: 0, dy: j ? 17 : 0 }, [l])))
    root.appendChild(title)
    root.appendChild(svg('rect', { class: 'band', x: pad.l, y: y(band[1]), width: W - pad.l - pad.r, height: y(band[0]) - y(band[1]) }))
    const yt = [-10, 0, 10, 20].filter((v) => v >= ylim[0] && v <= ylim[1])
    for (const v of yt) {
      root.appendChild(svg('line', { class: v === 0 ? 'zero' : 'grid', x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v) }))
      root.appendChild(svg('text', { class: 'muted label', x: pad.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, [String(v)]))
    }
    for (const o of onsets) root.appendChild(svg('text', { class: 'muted label', x: x(o), y: H - pad.b + 16, 'text-anchor': 'middle' }, [String(o)]))
    root.appendChild(svg('text', { class: 'muted label', x: (pad.l + W - pad.r) / 2, y: H - 4, 'text-anchor': 'middle' }, ['Hint onset (trial)']))
    for (const [a, label] of ['bayesian', 'centaur', 'human'].map((k) => AGENTS.find((p) => p[0] === k))) {
      const v = row[a]
      if (!v || v.some((p) => p[0] === null)) continue
      root.appendChild(svg('polyline', { class: `line ${CLS[a]} ${DASH[a]}`, points: onsets.map((o, i) => `${x(o)},${y(v[i][0])}`).join(' ') }))
      onsets.forEach((o, i) => {
        root.appendChild(svg('line', { class: `err ${CLS[a]}`, x1: x(o), x2: x(o), y1: y(v[i][1]), y2: y(v[i][2]) }))
        root.appendChild(marker(CLS[a], x(o), y(v[i][0]), a, a === 'human' ? 5 : 4))
        const hit = svg('rect', { class: 'hit', x: x(o) - 8, y: y(v[i][0]) - 8, width: 16, height: 16 })
        hit.dataset.tip = `<b>${label}</b>, onset ${o}<br>${fmt(v[i][0])} [${fmt(v[i][1])}, ${fmt(v[i][2])}]`
        root.appendChild(hit)
        if (a === 'human' && row.stars[i]) root.appendChild(svg('text', { class: 'star', x: x(o), y: y(v[i][2]) - 4, 'text-anchor': 'middle' }, [row.stars[i]]))
      })
    }
    return root
  }
  function renderTiming(type) {
    const d = DATA.timing.effects[type]
    let lo = -8, hi = 8
    for (const r of d.hints) for (const [a] of AGENTS) for (const p of r[a] || []) if (p[0] !== null) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[2]) }
    const ylim = [Math.floor(lo / 5) * 5, Math.ceil(hi / 5) * 5 + 2]
    const cells = d.hints.map(() => h('div', { class: 'chart' }))
    $('timing-chart').replaceChildren(h('div', { class: 'panels' }, cells), legend(AGENTS.map(([a, l]) => [CLS[a], l])))
    const pw = cw(cells[0], 150, 480), nLines = Math.max(...d.hints.map((r) => wrap(exampleHint(r.hint), Math.floor(pw / 7.1)).length))
    d.hints.forEach((r, i) => cells[i].appendChild(timingPanel(r, d.human_band, ylim, pw, nLines)))
    cur.timing = type
  }

  // ---------- Bayesian demo: the hierarchical agent of bayesian/hierarchical.py in miniature.
  // Every trial the agent sees the same evidence (button, points, and a shape or feature list when
  // the task shows one); each of the eight type backends scores that history analytically, P(type)
  // follows, and worlds are sampled type by type for Thompson sampling. A hint keeps the worlds it is true in.
  const PRI = { mu0: 45, s0: 20, sd: 5, epsMean: 2, epsSd: 1, lamMean: 0.2, lamSd: 0.08, ks: [1, 2, 3, 4, 5], thetaVar: 100 }
  const demo = { N: 3000 }
  const DEMO_HINTS = [
    ['none', 'No hint'], ['{max2} is the best.', '{max2} is the best.'], ['{min2} is the worst.', '{min2} is the worst.'],
    ['{max1} is better than {min2}, {min1}, and {mid}.', '{max1} is better than {min2}, {min1}, and {mid}.'],
  ]
  const LN2PI = Math.log(2 * Math.PI)
  const logpdf = (r, mu, v) => -0.5 * (LN2PI + Math.log(v) + ((r - mu) ** 2) / v)
  const logsumexp = (xs) => { const m = Math.max(...xs); return m === -Infinity ? -Infinity : m + Math.log(xs.reduce((s, x) => s + Math.exp(x - m), 0)) }
  const quantiles = (n, mean, sd) => Array.from({ length: n }, (_, i) => Math.max(mean + sd * probit((i + 0.5) / n), 1e-6))
  function probit(p) {  // inverse normal CDF (Acklam)
    const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924]
    const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857]
    const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878]
    const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1) }
    if (p > 1 - 0.02425) return -probit(1 - p)
    const q = p - 0.5, r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  }
  const nnStep = (mu, s2, a, r) => {  // one Normal-Normal update on arm a, returning the predictive log density
    if (Number.isNaN(r)) return -Infinity
    const v = PRI.sd * PRI.sd, ll = logpdf(r, mu[a], s2[a] + v), prec = 1 / s2[a] + 1 / v
    mu[a] = (mu[a] / s2[a] + r / v) / prec
    s2[a] = 1 / prec
    return ll
  }
  const fresh = () => ({ mu: Array(5).fill(PRI.mu0), s2: Array(5).fill(PRI.s0 * PRI.s0) })
  // the eight backends: log P(history | type) plus what Thompson sampling needs from each
  const BACKENDS = {
    stationary(H) { const b = fresh(); let ll = 0; H.a.forEach((a, t) => { ll += nnStep(b.mu, b.s2, a, H.o[t][a]) }); return { ll } },
    drifting(H) {
      const lls = quantiles(10, PRI.epsMean, PRI.epsSd).map((eps) => {
        const b = fresh(); let ll = 0
        H.a.forEach((a, t) => { ll += nnStep(b.mu, b.s2, a, H.o[t][a]); b.s2 = b.s2.map((s) => s + eps * eps) })
        return ll
      })
      return { ll: logsumexp(lls) - Math.log(lls.length) }
    },
    time_delayed(H) {
      const T = H.a.length, terms = []
      for (const k of PRI.ks) {
        if (k > T - 1) continue
        const b = fresh(); let ll = 0
        for (let t = k; t < T; t++) ll += nnStep(b.mu, b.s2, H.a[t - k], H.o[t][H.a[t - k]])
        terms.push(ll - Math.log(PRI.ks.length))
      }
      return { ll: terms.length ? logsumexp(terms) : -Infinity }
    },
    patch_foraging(H) {
      const lls = quantiles(40, PRI.lamMean, PRI.lamSd).map((lam) => {
        const b = fresh(), cap = Array(5).fill(PRI.mu0); let ll = 0
        H.a.forEach((a, t) => {
          ll += nnStep(b.mu, b.s2, a, H.o[t][a])
          cap[a] = b.mu[a]
          for (let i = 0; i < 5; i++) {
            if (i === a) { b.mu[i] = Math.max(0, b.mu[i] * Math.exp(-lam)); b.s2[i] *= Math.exp(-2 * lam) }
            else { b.mu[i] = Math.min(cap[i], b.mu[i] * Math.exp(lam)); b.s2[i] = Math.min(b.s2[i] * Math.exp(2 * lam), PRI.s0 * PRI.s0) }
          }
        })
        return ll
      })
      return { ll: logsumexp(lls) - Math.log(lls.length) }
    },
    two_context(H) { return ctxBackend(H, 2) },
    three_context(H) { return ctxBackend(H, 3) },
    bernoulli(H) {
      const al = Array(5).fill(1), be = Array(5).fill(1); let ll = 0
      for (let t = 0; t < H.a.length; t++) {
        const a = H.a[t], r = H.o[t][a]
        if (r !== 0 && r !== 100) return { ll: -Infinity, al, be }
        const p = al[a] / (al[a] + be[a])
        if (r === 100) { ll += Math.log(p); al[a]++ } else { ll += Math.log(1 - p); be[a]++ }
      }
      return { ll, al, be }
    },
    linear_contextual(H) {
      const d = H.x ? 3 : 5, v = PRI.sd * PRI.sd
      let Lam = eye(d, 1 / PRI.thetaVar), bvec = Array(d).fill(0), ll = 0
      for (let t = 0; t < H.a.length; t++) {
        const r = H.o[t][H.a[t]]
        if (Number.isNaN(r)) return { ll: -Infinity, mean: Array(d).fill(0), cov: eye(d, PRI.thetaVar) }
        const x = H.x ? H.x[t][H.a[t]] : eye(5, 1)[H.a[t]], S = inv(Lam), mu = matvec(S, bvec)
        ll += logpdf(r, dot(mu, x), dot(x, matvec(S, x)) + v)
        for (let i = 0; i < d; i++) { bvec[i] += r * x[i] / v; for (let j = 0; j < d; j++) Lam[i][j] += x[i] * x[j] / v }
      }
      const S = inv(Lam)
      return { ll, mean: matvec(S, bvec), cov: S }
    },
  }
  function ctxBackend(H, n) {
    const bs = Array.from({ length: n }, fresh); let ll = 0
    H.a.forEach((a, t) => { const c = (H.c ? H.c[t] : t) % n; ll += nnStep(bs[c].mu, bs[c].s2, a, H.o[t][a]) })
    return { ll, bs }
  }
  const eye = (d, v) => Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? v : 0)))
  const dot = (u, v) => u.reduce((s, x, i) => s + x * v[i], 0)
  const matvec = (M, v) => M.map((row) => dot(row, v))
  function inv(M) {  // Gauss-Jordan, small symmetric positive matrices only
    const d = M.length, A = M.map((row, i) => [...row, ...eye(d, 1)[i]])
    for (let i = 0; i < d; i++) {
      let p = i; for (let k = i + 1; k < d; k++) if (Math.abs(A[k][i]) > Math.abs(A[p][i])) p = k
      ;[A[i], A[p]] = [A[p], A[i]]
      const piv = A[i][i]; for (let j = 0; j < 2 * d; j++) A[i][j] /= piv
      for (let k = 0; k < d; k++) if (k !== i) { const f = A[k][i]; for (let j = 0; j < 2 * d; j++) A[k][j] -= f * A[i][j] }
    }
    return A.map((row) => row.slice(d))
  }
  function chol(S) { const d = S.length, L = eye(d, 0); for (let i = 0; i < d; i++) for (let j = 0; j <= i; j++) { let s = S[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]; L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j] } return L }
  function mulberry(seed) { return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let z = Math.imul(seed ^ (seed >>> 15), 1 | seed); z = (z + Math.imul(z ^ (z >>> 7), 61 | z)) ^ z; return ((z ^ (z >>> 14)) >>> 0) / 4294967296 } }
  // Beta draw as a ratio of two gammas, each from a fixed standard normal (Wilson-Hilferty), so a world keeps its draw
  const gammaFromZ = (k, z) => { const c = 1 / (9 * k); return k * Math.max(1e-6, 1 - c + z * Math.sqrt(c)) ** 3 }
  const betaFromZ = (al, be, z1, z2) => { const x = gammaFromZ(al, z1), y = gammaFromZ(be, z2); return x / (x + y) }
  function demoReset() {
    demo.type = pick(TYPES)
    demo.revealed = false
    demo.env = sampleEnv(demo.type)
    demo.g = makeGame(demo.type, 'none', demo.env)
    demo.arms = demo.env.arms || null  // value hints only where one set of button values holds for the whole game
    demo.ranks = demo.arms ? ranks(demo.arms) : null
    demo.n = [0, 0, 0, 0, 0]
    demo.last = [null, null, null, null, null]
    demo.H = { a: [], o: [], c: demo.g.contexts ? [] : null, x: demo.g.feats ? [] : null }
    const rng = mulberry(Math.floor(Math.random() * 1e9))
    const zn = () => Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng())
    demo.U = Array.from({ length: demo.N }, rng)
    demo.Z = Array.from({ length: demo.N }, () => Array.from({ length: 10 }, zn))
    demo.state = null
    const dh = $('demo-hint'), keep = dh.value || 'none'
    dh.replaceChildren(...DEMO_HINTS.filter(([v]) => v === 'none' || demo.arms).map(([v, l]) => h('option', { value: v }, v === 'none' ? l : processHint(l, demo.arms))))
    dh.value = demo.arms ? keep : 'none'
    demoRender()
  }
  function demoPull(i) {
    const g = demo.g
    if (g.t >= g.nTrials) return
    if (demo.H.c) demo.H.c.push(g.contexts[g.t])
    if (demo.H.x) demo.H.x.push(g.feats[g.t])
    const r = g.step(i)
    const row = Array(5).fill(g.type === 'time_delayed' ? 0 : NaN)
    if (g.type === 'time_delayed') { const seq = g.choices; if (seq.length >= 3) row[seq[seq.length - 3]] = r } else row[i] = r
    demo.H.a.push(i); demo.H.o.push(row)
    demo.n[i]++
    demo.last[i] = r
    demoRender()
  }
  function hintOK(w) {
    const r = demo.ranks, k = $('demo-hint').value
    if (k === 'none' || !r) return true
    const argmax = w.indexOf(Math.max(...w)), argmin = w.indexOf(Math.min(...w))
    if (k.startsWith('{max2} is the best')) return argmax === r.max2
    if (k.startsWith('{min2} is the worst')) return argmin === r.min2
    return w[r.max1] > w[r.min2] && w[r.max1] > w[r.min1] && w[r.max1] > w[r.mid]
  }
  function demoInfer() {
    const H = demo.H, T = H.a.length
    const out = {}, lls = TYPES.map((k) => { out[k] = BACKENDS[k](H); return T ? out[k].ll : 0 })
    const Z = logsumexp(lls), typeP = lls.map((l) => (Z === -Infinity ? 1 / TYPES.length : Math.exp(l - Z)))
    const shared = fresh()  // the shared arm-mean posterior every non-contextual type samples from
    H.o.forEach((row) => row.forEach((r, a) => { if (!Number.isNaN(r)) nnStep(shared.mu, shared.s2, a, r) }))
    const nextC = (n) => (H.c ? demo.g.contexts[Math.min(demo.g.t, demo.g.nTrials - 1)] : T) % n
    const nextX = H.x ? demo.g.feats[Math.min(demo.g.t, demo.g.nTrials - 1)] : null
    const worlds = []
    const Llc = chol(out.linear_contextual.cov)
    for (let i = 0; i < demo.N; i++) {
      let u = demo.U[i], ti = 0
      while (ti < TYPES.length - 1 && u > typeP[ti]) { u -= typeP[ti]; ti++ }
      const k = TYPES[ti], z = demo.Z[i]
      let w
      if (k === 'two_context' || k === 'three_context') { const b = out[k].bs[nextC(k === 'two_context' ? 2 : 3)]; w = b.mu.map((m, a) => m + Math.sqrt(b.s2[a]) * z[a]) }
      else if (k === 'bernoulli') w = out[k].al.map((al, a) => 100 * betaFromZ(al, out[k].be[a], z[a], z[5 + a]))
      else if (k === 'linear_contextual') {
        const { mean } = out[k]
        const theta = mean.map((m, i) => m + Llc[i].reduce((s, l, j) => s + l * z[j], 0))
        w = nextX ? nextX.map((x) => dot(x, theta)) : theta
        worlds.push({ ti, w, theta })
        continue
      } else w = shared.mu.map((m, a) => m + Math.sqrt(shared.s2[a]) * z[a])
      worlds.push({ ti, w })
    }
    return { typeP, worlds }
  }
  function demoState() {
    const g = demo.g, { typeP, worlds } = demoInfer()
    let kept = worlds.filter((p) => hintOK(p.w))
    const accept = kept.length / worlds.length
    const noWorld = kept.length === 0  // as in the agent: if no world is consistent with the hint, the hint carries no weight
    if (noWorld) kept = worlds
    const counts = [0, 0, 0, 0, 0]
    for (const p of kept) counts[p.w.indexOf(Math.max(...p.w))] += 1 / kept.length
    // the axis follows the sampled worlds (1st to 99th percentile), so it zooms in as the belief narrows
    const vals = kept.flatMap((p) => p.w).sort((a, b) => a - b)
    const q = (f) => vals[Math.min(vals.length - 1, Math.floor(f * vals.length))] || 0
    const span = Math.max(q(0.99) - q(0.01), 10), step = [1, 2, 5, 10, 20, 25, 50].find((s) => span / s <= 6) || 50
    const lo = Math.floor((q(0.01) - 0.05 * span) / step) * step, hi = Math.ceil((q(0.99) + 0.05 * span) / step) * step
    const bins = 60, bw = (hi - lo) / bins
    // a value that many worlds share exactly (a button worth 0 under every linear contextual world) is a point
    // mass; it is drawn as a spike at that value and kept out of the smoothed histogram
    const atoms = LETTERS.map((_, i) => {
      const tally = new Map()
      for (const p of kept) tally.set(p.w[i], (tally.get(p.w[i]) || 0) + 1)
      return [...tally].filter(([, n]) => n >= 0.03 * kept.length).map(([v, n]) => ({ v, mass: n / kept.length }))
    })
    const hists = LETTERS.map((_, i) => {
      const hist = new Array(bins).fill(0), skip = new Set(atoms[i].map((o) => o.v))
      for (const p of kept) { if (skip.has(p.w[i])) continue; const b = Math.floor((p.w[i] - lo) / bw); if (b >= 0 && b < bins) hist[b]++ }
      for (let pass = 0; pass < 2; pass++) for (let b = 1; b < bins - 1; b++) hist[b] = (hist[b - 1] + hist[b] + hist[b + 1]) / 3
      return hist
    })
    const peak = Math.max(...hists.flat(), 1e-9)  // one height scale for all five ridges
    const means = LETTERS.map((_, i) => kept.reduce((s, p) => s + p.w[i], 0) / Math.max(kept.length, 1))
    // belief over the three weights, from the linear contextual worlds, when the task shows features
    let theta = null
    if (demo.H.x) {
      const lw = kept.filter((p) => p.theta)
      const tv = lw.flatMap((p) => p.theta).sort((a, b) => a - b), tq = (f) => tv[Math.min(tv.length - 1, Math.floor(f * tv.length))] || 0
      const tspan = Math.max(tq(0.99) - tq(0.01), 4), tstep = [1, 2, 5, 10, 20].find((s) => tspan / s <= 6) || 20
      const tlo = Math.floor((tq(0.01) - 0.05 * tspan) / tstep) * tstep, thi = Math.ceil((tq(0.99) + 0.05 * tspan) / tstep) * tstep, tbw = (thi - tlo) / bins
      const th = [0, 1, 2].map((j) => {
        const hist = new Array(bins).fill(0)
        for (const p of lw) { const b = Math.floor((p.theta[j] - tlo) / tbw); if (b >= 0 && b < bins) hist[b]++ }
        for (let pass = 0; pass < 2; pass++) for (let b = 1; b < bins - 1; b++) hist[b] = (hist[b - 1] + hist[b] + hist[b + 1]) / 3
        return hist
      })
      const tpeak = Math.max(...th.flat(), 1e-9)
      theta = { lo: tlo, hi: thi, step: tstep, heights: th.map((h) => h.map((c) => c / tpeak)), share: lw.length / Math.max(kept.length, 1), means: [0, 1, 2].map((j) => lw.reduce((s, p) => s + p.theta[j], 0) / Math.max(lw.length, 1)) }
    }
    return { typeP, counts, lo, hi, step, bins, heights: hists.map((h) => h.map((c) => c / peak)), atoms, means, theta, accept, noWorld }
  }
  function demoDraw(S) {
    const g = demo.g
    const W = cw($('demo-chart')), rowH = 44, pad = { l: 34, r: 12, t: 8, b: 44 }, H = pad.t + 5 * rowH + pad.b
    const x = lin(S.lo, S.hi, pad.l, W - pad.r)
    const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Belief about each button value' })
    const step = tickStep(S.lo, S.hi)
    for (let v = Math.ceil(S.lo / step) * step; v <= S.hi; v += step) {
      root.appendChild(svg('line', { class: 'grid', x1: x(v), x2: x(v), y1: pad.t, y2: H - pad.b }))
      root.appendChild(svg('text', { class: 'muted label', x: x(v), y: H - pad.b + 16, 'text-anchor': 'middle' }, [String(Math.round(v))]))
    }
    root.appendChild(svg('text', { class: 'muted label', x: (pad.l + W - pad.r) / 2, y: H - 4, 'text-anchor': 'middle' }, ['Button value']))
    const bw = (S.hi - S.lo) / S.bins
    LETTERS.forEach((L, i) => {
      const base = pad.t + (i + 1) * rowH - 4
      const pts = S.heights[i].map((c, b) => `${x(S.lo + (b + 0.5) * bw)},${base - c * (rowH * 0.95)}`)
      root.appendChild(svg('polygon', { class: 'fill-human', points: [`${x(S.lo)},${base}`, ...pts, `${x(S.hi)},${base}`].join(' ') }))
      root.appendChild(svg('polyline', { class: 'line human', points: pts.join(' '), 'stroke-width': 1.5 }))
      for (const o of S.atoms[i]) {  // point mass: a spike whose height is its share of the worlds
        const top = base - Math.max(0.12, o.mass) * (rowH * 0.95)
        root.appendChild(svg('line', { class: 'line human', x1: x(o.v), x2: x(o.v), y1: base, y2: top, 'stroke-width': 3 }))
        root.appendChild(svg('circle', { class: 'human', cx: x(o.v), cy: top, r: 3 }))
      }
      root.appendChild(svg('text', { class: 'label', x: pad.l - 10, y: base - 2, 'text-anchor': 'end', style: 'font-family:var(--mono)' }, [L]))
      if (demo.last[i] !== null) root.appendChild(svg('circle', { class: 'you', cx: x(Math.max(S.lo, Math.min(S.hi, demo.last[i]))), cy: base, r: 3 }))
      const hit = svg('rect', { class: 'hit', x: pad.l, y: base - rowH + 4, width: W - pad.l - pad.r, height: rowH })
      hit.dataset.tip = `<b>${L}</b> belief mean ${fmt(S.means[i], 0)}<br>${demo.n[i]} pull${demo.n[i] === 1 ? '' : 's'}`
      root.appendChild(hit)
    })
    $('demo-chart').replaceChildren(root)
    const tb = $('demo-theta')
    tb.hidden = !S.theta
    if (S.theta) {
      const T = S.theta, rowT = 36, padT = { l: 34, r: 12, t: 28, b: 44 }, HT = padT.t + 3 * rowT + padT.b
      const xt = lin(T.lo, T.hi, padT.l, W - padT.r)
      const rt = svg('svg', { viewBox: `0 0 ${W} ${HT}`, role: 'img', 'aria-label': 'Belief about the three weights' })
      rt.appendChild(svg('text', { class: 'muted label', x: 0, y: 12 }, [`Weights, from the linear contextual worlds (${fmt(100 * T.share, 0)}%)`]))
      const tstep = tickStep(T.lo, T.hi)
      for (let v = Math.ceil(T.lo / tstep) * tstep; v <= T.hi; v += tstep) {
        rt.appendChild(svg('line', { class: v === 0 ? 'zero' : 'grid', x1: xt(v), x2: xt(v), y1: padT.t, y2: HT - padT.b }))
        rt.appendChild(svg('text', { class: 'muted label', x: xt(v), y: HT - padT.b + 16, 'text-anchor': 'middle' }, [String(v)]))
      }
      rt.appendChild(svg('text', { class: 'muted label', x: (padT.l + W - padT.r) / 2, y: HT - 4, 'text-anchor': 'middle' }, ['Weight on each of the three numbers']))
      const tbw = (T.hi - T.lo) / S.bins
      T.heights.forEach((hs, j) => {
        const base = padT.t + (j + 1) * rowT - 4
        const pts = hs.map((c, b) => `${xt(T.lo + (b + 0.5) * tbw)},${base - c * (rowT * 0.95)}`)
        rt.appendChild(svg('polygon', { class: 'fill-bayes', points: [`${xt(T.lo)},${base}`, ...pts, `${xt(T.hi)},${base}`].join(' ') }))
        rt.appendChild(svg('polyline', { class: 'line bayes', points: pts.join(' '), 'stroke-width': 1.5 }))
        rt.appendChild(svg('text', { class: 'label', x: padT.l - 10, y: base - 2, 'text-anchor': 'end' }, [`θ${j + 1}`]))
        const hit = svg('rect', { class: 'hit', x: padT.l, y: base - rowT + 4, width: W - padT.l - padT.r, height: rowT })
        hit.dataset.tip = `<b>θ${j + 1}</b> mean ${fmt(T.means[j], 1)} · true ${[-2, 0, 5][j]}`
        rt.appendChild(hit)
      })
      tb.replaceChildren(rt)
    }
    const bars = (labels, vals, title, cls, H2 = 150, lines = 1) => {
      const tl = wrap(title, Math.floor(W / 7.8))  // a long title (the hint note) wraps instead of running off a narrow chart
      H2 += 17 * (tl.length - 1)
      const W2 = W, p2 = { l: 34, r: 12, t: 11 + 17 * tl.length, b: 24 + 17 * lines }, n = vals.length
      const x2 = lin(-0.5, n - 0.5, p2.l, W2 - p2.r), y2 = lin(0, 1, H2 - p2.b, p2.t)
      const r2 = svg('svg', { viewBox: `0 0 ${W2} ${H2}`, role: 'img', 'aria-label': title })
      const tt = svg('text', { class: 'muted label', x: 0, y: 12 })
      tl.forEach((l, j) => tt.appendChild(svg('tspan', { x: 0, dy: j ? 17 : 0 }, [l])))
      r2.appendChild(tt)
      r2.appendChild(svg('line', { class: 'axis', x1: p2.l, x2: W2 - p2.r, y1: y2(0), y2: y2(0) }))
      vals.forEach((v, i) => {
        const bw2 = (x2(1) - x2(0)) * 0.6
        const vv = Math.max(0, v)
        r2.appendChild(svg('rect', { class: 'bar ' + cls, x: x2(i) - bw2 / 2, y: y2(vv), width: bw2, height: y2(0) - y2(vv), rx: 2 }))
        const lab = svg('text', { class: 'muted label', x: x2(i), y: H2 - p2.b + 16, 'text-anchor': 'middle' })
        String(labels[i]).split('\n').forEach((l, j) => lab.appendChild(svg('tspan', { x: x2(i), dy: j ? 17 : 0 }, [l])))
        lab.appendChild(svg('tspan', { x: x2(i), dy: 17 }, [`${fmt(100 * v, 0)}%`]))
        r2.appendChild(lab)
      })
      return r2
    }
    const hbars = (labels, vals, title, cls) => {
      const rh = 26, p2 = { l: 158, r: 52, t: 28, b: 2 }, H2 = p2.t + vals.length * rh + p2.b
      const x2 = lin(0, 1, p2.l, W - p2.r)
      const r2 = svg('svg', { viewBox: `0 0 ${W} ${H2}`, role: 'img', 'aria-label': title })
      r2.appendChild(svg('text', { class: 'muted label', x: 0, y: 12 }, [title]))
      r2.appendChild(svg('line', { class: 'axis', x1: p2.l, x2: p2.l, y1: p2.t, y2: H2 - p2.b }))
      vals.forEach((v, i) => {
        const cy = p2.t + (i + 0.5) * rh, vv = Math.max(0, v)
        r2.appendChild(svg('text', { class: 'muted label', x: p2.l - 10, y: cy + 5, 'text-anchor': 'end' }, [labels[i]]))
        r2.appendChild(svg('rect', { class: 'bar ' + cls, x: p2.l, y: cy - 8, width: x2(vv) - p2.l, height: 16, rx: 2 }))
        r2.appendChild(svg('text', { class: 'muted label', x: W - 4, y: cy + 5, 'text-anchor': 'end' }, [`${fmt(100 * v, 0)}%`]))
      })
      return r2
    }
    const typeLabels = TYPES.map((k) => (demo.revealed && k === demo.type ? '✓ ' : '') + LABEL[k])
    $('demo-types').replaceChildren(hbars(typeLabels, S.typeP, demo.revealed ? `P(task type) · true task: ${LABEL[demo.type].toLowerCase()}` : 'P(task type)', 'bayes'))
    const hintNote = S.noWorld ? ', none fit the hint, so it is ignored' : S.accept < 1 ? `, ${fmt(100 * S.accept, 0)}% fit the hint` : ''
    $('demo-probs').replaceChildren(bars(LETTERS, S.counts, `P(choice) · ${demo.N} sampled worlds${hintNote}`, 'human'))
  }
  const mix = (a, b, f) => a + (b - a) * f
  function demoRender() {
    const g = demo.g
    const ctx = $('demo-context'), ft = $('demo-features')
    ctx.hidden = !g.contexts
    if (g.contexts) ctx.replaceChildren(shapeSVG(g.contexts[Math.min(g.t, g.nTrials - 1)]))
    ft.hidden = !g.feats
    if (g.feats) ft.replaceChildren(...g.feats[Math.min(g.t, g.nTrials - 1)].map((f) => h('span', {}, f.join(' '))))
    const armsEl = $('demo-arms')
    armsEl.replaceChildren()
    LETTERS.forEach((L, i) => {
      const b = h('button', { class: 'arm', onclick: () => demoPull(i), disabled: g.t >= g.nTrials ? '' : null }, [L])
      b.title = demo.n[i] ? `${demo.n[i]} pulls, last ${fmt(demo.last[i], 0)}` : 'Not tried yet'
      armsEl.appendChild(b)
    })
    $('demo-trial').textContent = `Trial ${Math.min(g.t + 1, g.nTrials)} / ${g.nTrials}`
    const target = demoState(), from = demo.state
    cancelAnimationFrame(demo.anim)
    if (!from || from.bins !== target.bins || matchMedia('(prefers-reduced-motion: reduce)').matches) { demo.state = target; demoDraw(target); return }
    // tween every number from the last drawn state to the new one, so beliefs slide instead of jumping
    const t0 = performance.now(), dur = 450
    const frame = (now) => {
      const f = Math.min(1, (now - t0) / dur), e = 1 - (1 - f) ** 3
      // ridges are resampled on the target axis; blend the axis itself so a zoom glides
      const S = {
        ...target,
        lo: mix(from.lo, target.lo, e), hi: mix(from.hi, target.hi, e),
        typeP: target.typeP.map((v, i) => mix(from.typeP[i], v, e)),
        counts: target.counts.map((v, i) => mix(from.counts[i], v, e)),
        heights: target.heights.map((hs, i) => hs.map((v, b) => mix(from.heights[i][b], v, e))),
        theta: target.theta && from.theta ? { ...target.theta, lo: mix(from.theta.lo, target.theta.lo, e), hi: mix(from.theta.hi, target.theta.hi, e), heights: target.theta.heights.map((hs, j) => hs.map((v, b) => mix(from.theta.heights[j][b], v, e))) } : target.theta,
      }
      demoDraw(S)
      if (f < 1) demo.anim = requestAnimationFrame(frame)
      else demo.state = target
    }
    demo.anim = requestAnimationFrame(frame)
  }

  // ---------- Centaur: the prompt it reads and the choices it drew, replayed from one stored simulated game
  const MAIN_TEXT = 'You will play a game in which you have to make choices between five buttons labeled 1, 2, 3, 4, and 5. When you choose each button, it will give back some number of points from 0 to 100, but with some randomness around its number. In each game, you will have 20 or more choices to make to get as many points as you can. Finally, in some games, you will receive a hint which will help you in the game. In others, you will be given no hint. This is a brand new game, and the button values have been reset.'
  const BERNOULLI_TEXT = 'You will play a game in which you have to make choices between five buttons labeled 1, 2, 3, 4, and 5. When you choose each button, it will give back 0 or 100 points based on some probability associated with that button, which will not change over the course of the game. In each game, you will have 40 choices to make to get as many points as you can. Finally, in some games, you will receive a hint which will help you in the game. In others, you will be given no hint. This is a brand new game, and the button values have been reset.'
  const LINCON_TEXT = 'You will play a game in which you have to make choices between five buttons labeled 1, 2, 3, 4, and 5. Each button is associated with a vector of three numbers, which are listed underneath the button. The reward you receive from each button is a weighted sum of these numbers, where the weights are fixed for the duration of the game. In each game, you will have 20 choices to make to get as many points as you can. Finally, in some games, you will receive a hint which will help you in the game. In others, you will be given no hint. This is a brand new game, and the button weights have been reset.'
  const QUESTION = 'Which button should you choose next? Your options are 1, 2, 3, 4, or 5. Your choice is <<'
  let centaurSel = { type: 'stationary', hint: 'none' }
  const centaurEx = () => byHint(DATA.centaur[centaurSel.type], centaurSel.hint)
  // the stored prompt names buttons 1 to 5; show the same hint with the letters used everywhere else on the page
  const centaurLetters = (ex, hint) => (hint.includes('{') && ex ? ex.hint.replace(/\b([1-5])\b/g, (m, d) => LETTERS[+d - 1]) : /choose [ASDFG]\b/.test(hint) || !ex ? hint : ex.hint)
  function centaurHintOptions(type, value) {
    const sel = $('centaur-hint')
    sel.replaceChildren(h('option', { value: 'none' }, 'No hint'))
    for (const hint of HINTS[type]) sel.appendChild(h('option', { value: hint }, centaurLetters(byHint(DATA.centaur[type], hint), hint)))
    sel.value = value
  }
  function numberHint(hint, arms) {
    let out = hint
    if (arms) { const r = ranks(arms); out = out.replace(/{(\w+)}/g, (m, k) => (r[k] === undefined ? m : String(r[k] + 1))) }
    return out.replace(/choose ([ASDFG])\b/g, (m, L) => 'choose ' + (LETTERS.indexOf(L) + 1))
  }
  function replayPoints(ex, type) {  // the logs keep arms and choices but not the drawn points, so draw them again from the same arm values
    const draw = (t, c) => {
      const arms = ex.arms[t], v = arms[c]
      if (type === 'bernoulli') return Math.random() < (v > 1 ? v / 100 : v) ? 100 : 0
      if (type === 'linear_contextual') return v
      return v + 5 * gauss()
    }
    const pts = ex.choice.map((c, t) => draw(t, c))
    if (type === 'time_delayed') return pts.map((p, t) => (t < 2 ? 0 : pts[t - 2]))
    return pts
  }
  function renderPrompt() {
    const { type, hint } = centaurSel
    const ex = centaurEx()
    const ctxText = type === 'two_context' ? ' You will also see either a square or a circle.' : type === 'three_context' ? ' You will also see either a square, a circle, or a triangle.' : ''
    const base = type === 'bernoulli' ? BERNOULLI_TEXT : type === 'linear_contextual' ? LINCON_TEXT : MAIN_TEXT + ctxText
    const pre = $('prompt')
    pre.replaceChildren(base + '\n\nHint: ', h('i', {}, ex ? ex.hint : hint === 'none' ? 'No hint.' : numberHint(hint, null)), '\n')
    if (!ex) { pre.append(QUESTION); return }
    const n = rp.i, T = ex.probs.length, isCtx = ex.context && ex.context[0] >= 0
    if (isCtx) pre.append(`You see a ${SHAPES[ex.context[Math.min(n, T - 1)]]}.\n`)
    for (let i = 0; i < n; i++) {
      const c = ex.choice[i], r = rp.points[i].toFixed(2)
      pre.append(isCtx ? `Trial ${i}: shape: ${SHAPES[ex.context[i]]}, choice: ` : `Trial ${i}: choice: `, h('b', {}, `<<${c + 1}>>`), `, points: ${r}.\n`)
    }
    if (n < T) pre.append(QUESTION)
    else pre.append(h('span', { class: 'muted' }, 'Game over.'))
    pre.scrollTop = pre.scrollHeight
  }
  // step-by-step replay: its choice probabilities at each trial and the button it drew
  const rp = { i: 0, timer: null, points: [] }
  function replayStop() { clearInterval(rp.timer); rp.timer = null }
  function replayStep() {
    const ex = centaurEx()
    if (rp.i >= ex.probs.length) { replayStop(); return }
    rp.i++
    if (rp.i >= ex.probs.length) replayStop()
    renderReplay()
    renderPrompt()
  }
  function replayRun() {
    if (rp.timer) { replayStop(); renderReplay(); return }
    const ex = centaurEx()
    if (rp.i >= ex.probs.length) replayReset(ex)
    rp.timer = setInterval(replayStep, 450)
    renderReplay()
  }
  function renderReplay() {
    const ex = centaurEx()
    const box = $('centaur-chart')
    if (!ex) { box.replaceChildren(h('p', { class: 'sub' }, 'Centaur was not run with this hint.')); return }
    const T = ex.probs.length, i = rp.i, cur = Math.min(i, T - 1), done = i >= T
    const probs = ex.probs[cur], chosen = i > 0 ? ex.choice[i - 1] : null
    const top = h('div', { class: 'game-top' }, [
      h('span', {}, i === 0 ? `${T} trials` : `Trial ${Math.min(i, T)} / ${T}`),
      h('span', {}, i > 0 ? `${ex.choice.slice(0, i).filter((c, k) => c === ex.optimal[k]).length} best-button picks` : ''),
    ])
    const hintLine = centaurSel.hint === 'none' ? null : h('div', { class: 'hint-box' }, centaurLetters(ex, centaurSel.hint))
    const ctx = h('div', { class: 'context' })
    const isCtx = ex.context && ex.context[0] >= 0
    ctx.hidden = !isCtx
    if (isCtx) ctx.appendChild(shapeSVG(ex.context[i > 0 ? i - 1 : 0]))
    const arms = h('div', { class: 'replay-arms' }, LETTERS.map((L, a) => {
      const p = i > 0 ? ex.probs[i - 1][a] : probs[a]
      const bar = h('div', { class: 'pbar' + (chosen === a ? ' chosen' : '') }, [h('div', { class: 'pfill', style: `height:${Math.round(100 * p)}%` })])
      return h('div', { class: 'replay-arm' + (chosen === a ? ' chosen' : ''), 'data-tip': `<b>${L}</b> ${fmt(100 * p, 0)}%` }, [bar, h('span', { class: 'mono' }, L), h('span', { class: 'pct' }, `${fmt(100 * p, 0)}%`)])
    }))
    const hist = h('ol', { class: 'history replay-history' }, ex.choice.slice(0, i).map((c, k) => h('li', { class: c === ex.optimal[k] ? 'best' : '' }, [h('b', {}, LETTERS[c])])))
    const actions = h('div', { class: 'game-actions' }, [
      h('button', { class: 'btn', onclick: replayRun }, rp.timer ? 'Pause' : done ? 'Replay' : i === 0 ? 'Run' : 'Resume'),
      h('button', { class: 'btn ghost', onclick: () => { replayStop(); replayStep() }, disabled: done ? '' : null }, 'Step'),
    ])
    box.replaceChildren(h('div', { class: 'game replay' }, [top, hintLine, ctx, arms, hist, actions].filter(Boolean)))
  }
  function replayReset(ex) {
    rp.i = 0
    rp.points = ex ? replayPoints(ex, centaurSel.type) : []
    renderPrompt()
  }
  function renderCentaur() {
    replayStop()
    replayReset(centaurEx())
    renderReplay()
  }

  // ---------- IBT: an animated game of each task, showing the true button values a player never sees
  const cur = {}
  const IBT_NOTE = {
    stationary: ' · button values stay fixed',
    drifting: ' · button values drift every trial',
    patch_foraging: ' · used buttons deplete, rested ones recover',
    two_context: ' · the shape sets the button values',
    three_context: ' · the shape sets the button values',
    bernoulli: ' · a press pays 100 or 0',
    linear_contextual: ' · points = −2·first + 0·second + 5·third',
  }
  const ib = { timer: null, g: null, env: null, type: 'stationary' }
  function ibtValues(g) {
    const t = Math.max(g.t - 1, 0)
    if (g.type === 'linear_contextual') return g.feats[t].map((f) => f.reduce((s, v, i) => s + v * g.theta[i], 0))
    if (g.type === 'two_context' || g.type === 'three_context') return g.armsAt(t)
    return g.arms
  }
  function ibtDraw() {
    const g = ib.g, type = g.type, W = cw($('ibt-anim'), 440, 640), H = 264, pad = { l: type === 'bernoulli' ? 52 : 36, r: 12, t: 40, b: 60 }
    const root = svg('svg', { class: 'anim', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Button values over a game' })
    const last = g.t - 1, chosen = last >= 0 ? g.choices[last] : null
    const lastReward = last >= 0 ? g.rewards[last] : null
    const rewardArm = type === 'time_delayed' ? (ib.delivered ? ib.delivered.arm : null) : chosen
    const isCtx = type === 'two_context' || type === 'three_context'
    const title = (g.t === 0 ? 'Start' : `Trial ${g.t} / ${g.nTrials}`) + (IBT_NOTE[type] || '')
    root.appendChild(svg('text', { class: 'muted label', x: 0, y: 14 }, [title]))
    if (type === 'linear_contextual') {
      const colW = (W - pad.l - pad.r) / 5, cw = colW * 0.88, top = pad.t + 24, ch = 96
      const feats = g.feats[Math.max(g.t - 1, 0)], vals = ibtValues(g)
      LETTERS.forEach((L, a) => {
        const cx = pad.l + colW * (a + 0.5), on = chosen === a
        root.appendChild(svg('rect', { class: 'card' + (on ? ' on' : ''), x: cx - cw / 2, y: top, width: cw, height: ch, rx: 6 }))
        root.appendChild(svg('text', { class: 'label', x: cx, y: top + 22, 'text-anchor': 'middle', style: 'font-family:var(--mono)' }, [L]))
        root.appendChild(svg('text', { class: 'feat' + (on ? ' on' : ''), x: cx, y: top + 66, 'text-anchor': 'middle' }, [feats[a].join(' ')]))
        if (on && lastReward !== null) root.appendChild(svg('text', { class: 'reward label', x: cx, y: top + ch + 22, 'text-anchor': 'middle' }, [`= ${fmt(lastReward, 0)} points`]))
        else root.appendChild(svg('text', { class: 'muted label', x: cx, y: top + ch + 22, 'text-anchor': 'middle' }, [`= ${fmt(vals[a], 0)}`]))
      })
    } else if (type === 'bernoulli') {
      const vals = ibtValues(g), y = lin(0, 100, H - pad.b, pad.t + 10)
      for (const v of [0, 50, 100]) {
        root.appendChild(svg('line', { class: 'grid', x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v) }))
        root.appendChild(svg('text', { class: 'muted label', x: pad.l - 8, y: y(v) + 4, 'text-anchor': 'end' }, [v + '%']))
      }
      const colW = (W - pad.l - pad.r) / 5, bw = colW * 0.56
      LETTERS.forEach((L, a) => {
        const cx = pad.l + colW * (a + 0.5), v = vals[a], on = chosen === a
        root.appendChild(svg('rect', { class: 'bar' + (on ? ' on' : ''), x: cx - bw / 2, y: y(v), width: bw, height: y(0) - y(v), rx: 3 }))
        root.appendChild(svg('text', { class: 'label', x: cx, y: H - pad.b + 20, 'text-anchor': 'middle', style: 'font-family:var(--mono)' }, [L]))
        if (on && lastReward !== null) {
          const win = lastReward > 0, coin = svg('g', { class: 'coin' + (win ? ' win' : '') })
          coin.appendChild(svg('circle', { cx, cy: y(v) - 24, r: 16 }))
          coin.appendChild(svg('text', { x: cx, y: y(v) - 20, 'text-anchor': 'middle' }, [win ? '100' : '0']))
          root.appendChild(coin)
        }
      })
    } else {
      const means = ibtValues(g), yb = H - pad.b, peak = 120, sd = 5
      // drifting values can leave 0 to 100: the axis widens in steps of 25 so no curve runs off it
      const mn = Math.min(...means.map((m) => m - 3 * sd), ...ib.samples), mx = Math.max(...means.map((m) => m + 3 * sd), ...ib.samples)
      if (mn < ib.dom[0]) ib.dom[0] = Math.floor(mn / 25) * 25 - 15
      if (mx > ib.dom[1]) ib.dom[1] = Math.ceil(mx / 25) * 25 + 15
      const x = lin(ib.dom[0], ib.dom[1], pad.l, W - pad.r)
      for (let v = Math.ceil(ib.dom[0] / 25) * 25; v <= ib.dom[1]; v += 25) root.appendChild(svg('text', { class: 'muted label', x: x(v), y: yb + 34, 'text-anchor': 'middle' }, [String(v)]))
      const curve = (m) => { const pts = []; for (let v = m - 3 * sd; v <= m + 3 * sd + 1e-9; v += 0.75) pts.push(`${x(v)},${yb - peak * Math.exp(-((v - m) ** 2) / (2 * sd * sd))}`); return pts }
      const order = LETTERS.map((_, a) => a).sort((a, b) => (a === chosen) - (b === chosen))
      for (const a of order) {
        const m = means[a], pts = curve(m), on = chosen === a ? ' on' : ''
        root.appendChild(svg('polygon', { class: 'dens' + on, points: [`${x(m - 3 * sd)},${yb}`, ...pts, `${x(m + 3 * sd)},${yb}`].join(' ') }))
        root.appendChild(svg('polyline', { class: 'curve' + on, points: pts.join(' ') }))
      }
      root.appendChild(svg('line', { class: 'base', x1: pad.l, x2: W - pad.r, y1: yb, y2: yb }))
      // button letters sit under their curves and are nudged apart when curves overlap
      const labs = LETTERS.map((_, a) => ({ a, px: x(means[a]) })).sort((p, q) => p.px - q.px)
      for (let i = 1; i < 5; i++) labs[i].px = Math.max(labs[i].px, labs[i - 1].px + 16)
      labs[4].px = Math.min(labs[4].px, W - pad.r - 6)
      for (let i = 3; i >= 0; i--) labs[i].px = Math.min(labs[i].px, labs[i + 1].px - 16)
      for (const { a, px } of labs) root.appendChild(svg('text', { class: 'label' + (chosen === a ? ' onletter' : ''), x: px, y: yb + 16, 'text-anchor': 'middle', style: 'font-family:var(--mono)' }, [LETTERS[a]]))
      ib.samples.forEach((s, i) => root.appendChild(svg('circle', { class: 'sample', cx: x(s), cy: yb, r: 3.5, opacity: (i + 1) / (ib.samples.length + 1) })))
      if (lastReward !== null && rewardArm !== null) {
        root.appendChild(svg('circle', { class: 'sample on', cx: x(lastReward), cy: yb, r: 4.5 }))
        root.appendChild(svg('text', { class: 'reward label', x: x(lastReward), y: yb - peak - 10, 'text-anchor': 'middle' }, [`+${fmt(lastReward, 0)}`]))
        root.appendChild(svg('line', { class: 'drop', x1: x(lastReward), x2: x(lastReward), y1: yb - peak - 4, y2: yb - 6 }))
      }
      if (type === 'time_delayed') {
        const sw = 78, sh = 30, gap = 30, x0 = W - pad.r - 3 * sw - 2 * gap, y0 = 24
        root.appendChild(svg('text', { class: 'muted label', x: x0, y: y0 - 6 }, ['pressed → waiting → waiting → paid']))
        const cells = [ib.queue[1], ib.queue[0], ib.delivered]
        cells.forEach((s, k) => {
          const bx = x0 + k * (sw + gap), paid = k === 2
          root.appendChild(svg('rect', { class: 'slot' + (paid && s ? ' paid' : ''), x: bx, y: y0, width: sw, height: sh, rx: 4 }))
          if (s) root.appendChild(svg('text', { class: 'label' + (paid ? ' onlabel' : ''), x: bx + sw / 2, y: y0 + 20, 'text-anchor': 'middle', style: 'font-family:var(--mono)' }, [`${LETTERS[s.arm]} ${fmt(s.r, 0)}`]))
          if (k < 2) root.appendChild(svg('text', { class: 'muted label', x: bx + sw + gap / 2, y: y0 + 20, 'text-anchor': 'middle' }, ['→']))
        })
      }
    }
    if (isCtx) {
      const ctx = g.contexts[Math.max(g.t - 1, 0)], s = shapeSVG(ctx, 40)
      s.setAttribute('x', W - pad.r - 40); s.setAttribute('y', 22)
      root.appendChild(svg('text', { class: 'label', x: W - pad.r - 48, y: 48, 'text-anchor': 'end' }, [`You see a ${SHAPES[ctx]}`]))
      root.appendChild(s)
    }
    root.appendChild(svg('text', { class: 'muted label', x: W - pad.r, y: H - 4, 'text-anchor': 'end' }, ['True button distributions hidden']))
    $('ibt-anim').replaceChildren(root)
  }
  function ibtTick() {
    const g = ib.g
    if (g.t >= g.nTrials) { ib.g = makeGame(ib.type, 'none', ib.env); ib.queue = []; ib.delivered = null; ib.samples = []; ib.dom = [-15, 115]; ibtDraw(); return }
    const prev = g.choices[g.choices.length - 1]
    let c
    if (g.type === 'linear_contextual') { const v = g.feats[g.t].map((f) => f.reduce((s, x, i) => s + x * g.theta[i], 0)); c = Math.random() < 0.6 ? v.indexOf(Math.max(...v)) : Math.floor(Math.random() * 5) }
    else c = prev !== undefined && Math.random() < 0.65 ? prev : Math.floor(Math.random() * 5)
    if (g.type === 'time_delayed') {  // draw the reward now and hold it in a queue of two, as the task's delay does
      ib.queue.push({ arm: c, r: g.arms[c] + 5 * gauss() })
      ib.delivered = ib.queue.length > 2 ? ib.queue.shift() : null
      g.choices.push(c); g.rewards.push(ib.delivered ? ib.delivered.r : 0); g.t += 1
    } else g.step(c)
    const r = g.rewards[g.t - 1]
    if (g.type !== 'bernoulli' && g.type !== 'linear_contextual' && (g.type !== 'time_delayed' || ib.delivered)) ib.samples = [...ib.samples.slice(-5), r]
    ibtDraw()
  }
  function renderIBT(type) {
    clearInterval(ib.timer)
    ib.type = type
    ib.env = sampleEnv(type)
    ib.g = makeGame(type, 'none', ib.env)
    ib.queue = []; ib.delivered = null; ib.samples = []; ib.dom = [-15, 115]
    const arms = labelArms(ib.env)
    $('ibt-hints').replaceChildren(h('li', {}, 'No hint.'), ...HINTS[type].map((hint) => h('li', {}, arms ? processHint(hint, arms) : hint)))
    ibtDraw()
    ib.timer = setInterval(ibtTick, 700)
  }

  // ---------- boot
  async function boot() {
    await load()
    const taskItems = TYPES.map((t) => [t, LABEL[t]])
    chips($('ibt-chips'), taskItems, 'stationary', renderIBT)
    renderIBT('stationary')
    chips($('task-chips'), taskItems, taskSel, (k) => { taskSel = k; game = null; env = sampleEnv(k); hintOptions(k); $('result').hidden = true; renderGame() })
    hintOptions(taskSel)
    $('hint-select').addEventListener('change', (e) => { hintSel = e.target.value })
    $('start-btn').addEventListener('click', start)
    renderGame()
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'SELECT' || e.metaKey || e.ctrlKey) return
      const i = LETTERS.indexOf(e.key.toUpperCase())
      if (i < 0 || e.altKey) return
      // the letter goes to whichever game's buttons sit nearest the middle of the window
      const off = (el) => { const r = el.getBoundingClientRect(); return r.bottom < 0 || r.top > innerHeight ? Infinity : Math.abs((r.top + r.bottom) / 2 - innerHeight / 2) }
      const people = game && game.t < game.nTrials ? off($('arms')) : Infinity
      const agent = demo.g && demo.g.t < demo.g.nTrials ? off($('demo-arms')) : Infinity
      if (people === Infinity && agent === Infinity) return
      e.preventDefault()
      if (people <= agent) choose(i)
      else demoPull(i)
    })
    chips($('centaur-chips'), taskItems, centaurSel.type, (k) => { centaurSel = { type: k, hint: 'none' }; centaurHintOptions(k, 'none'); renderCentaur() })
    centaurHintOptions(centaurSel.type, 'none')
    $('centaur-hint').addEventListener('change', (e) => { centaurSel.hint = e.target.value; renderCentaur() })
    renderCentaur()
    chips($('curve-chips'), taskItems, 'stationary', renderCurves)
    renderCurves('stationary')
    chips($('effect-chips'), taskItems, 'stationary', renderEffects)
    renderEffects('stationary')
    chips($('timing-chips'), [['stationary', 'Stationary'], ['patch_foraging', 'Patch foraging']], 'stationary', renderTiming)
    renderTiming('stationary')
    $('demo-reveal').addEventListener('click', () => { demo.revealed = true; demoRender() })
    $('demo-hint').addEventListener('change', demoRender)
    $('demo-reset').addEventListener('click', demoReset)
    demoReset()
    for (const id of ['result-chart', 'curve-chart', 'effect-chart', 'timing-chart', 'demo-chart', 'demo-theta', 'centaur-chart']) bindTips($(id))
    let lastW = innerWidth, rz = null
    addEventListener('resize', () => {
      clearTimeout(rz)
      rz = setTimeout(() => {
        if (innerWidth === lastW) return
        lastW = innerWidth
        renderCurves(cur.curves); renderEffects(cur.effects); renderTiming(cur.timing); ibtDraw()
        if (demo.state) demoDraw(demo.state)
      }, 150)
    })
    $('copy-cite').addEventListener('click', () => {
      const b = $('copy-cite')
      navigator.clipboard.writeText($('cite').textContent).then(() => { b.textContent = 'Copied' }).catch(() => { window.getSelection().selectAllChildren($('cite')) })
    })
  }
  boot()
})()
