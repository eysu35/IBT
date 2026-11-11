# website

Static project page for the IBT paper: the three learners (playable task, Bayesian belief demo, Centaur
prompt and choice probabilities) and interactive versions of Figs 3–5 and 7. Plain HTML/CSS/JS, no build step.

```
website/
├── index.html      # page
├── style.css       # tokens, layout (light only)
├── app.js          # task simulation (port of behavioral/.../bandits.js), charts, Bayesian demo
├── build_data.py   # exports data/*.json from the repo's tables and parquets
├── preview.html    # source of preview.png, the 1200x630 link-preview image (og:image)
└── data/           # curves, hint_effects, timing, centaur (example games with per-trial probs)
```

## Preview

```bash
python3 -m http.server -d website 8000   # http://localhost:8000
```

## Refresh the data

```bash
python3 website/build_data.py
```

Reads `analysis/tables/{hint_effects,timing}.csv`, the two human `hint_effects.csv` files (significance stars) and the
human/Bayesian/Centaur trial parquets (Centaur rows carry per-trial choice probabilities). Commit the regenerated `data/`.

## Hosting

GitHub Pages from `.github/workflows/pages.yml` (deploys `website/` on every push to `main` that touches
it). One-time setup on GitHub: Settings → Pages → Source: GitHub Actions. The site is then at
`https://eysu35.github.io/IBT/`. For a custom domain, add a `CNAME` file in `website/` and the DNS record.

Optional: the original Smile experiment can be deployed in presentation mode
(branch `presentation` in the experiment repo → exps.gureckislab.org) and linked from the Play section.
