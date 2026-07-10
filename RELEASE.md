# RELEASE.md — Sundial publish checklist

This is the human-run checklist to take Sundial from a local, never-published
repo to a live Show HN. Nothing here has been executed — the whole M2 line was
built "launch-ready, nothing published." Work top to bottom; each step is a
checkbox. Commands assume you are in the repo root on the branch you intend to
ship (merge `m2-launch-ready` to `main` first if you haven't).

Order matters: get the repo public and CI green first, then the demo deploy
(you need its URL), then npm, then fill the URL into the writeup, then post.

---

## 0. Decisions to make before you touch anything

- [ ] **Repo name.** The product, the Cargo crates, and the npm package are all
  "sundial" (`crates/sundial-*`, package `sundial-lp`); only the local working
  directory is `or-fable`. Recommendation: name the public GitHub repo
  **`sundial`** so the repo, product, crates, and package all agree. If you
  want the org/handle in front, `<you>/sundial`. Decide now — it appears in the
  clone URL, the Pages URL, and every link in the post.
- [ ] **Visibility.** Public. (It's a Show HN.)
- [ ] **License is already MIT OR Apache-2.0** — confirm `LICENSE-MIT` and
  `LICENSE-APACHE` (or equivalent) are present at the root before going public;
  add them if missing, since `Cargo.toml` already declares the dual license.
- [ ] **Merge to `main`.** Confirm `m2-launch-ready` is merged (or is the branch
  you're shipping) and the tree is green locally:
  ```bash
  bash scripts/verify_clean_checkout.sh   # fmt + workspace tests + wasm + web build
  ```

## 1. Create the GitHub repo and push

- [ ] Create the empty public repo (no README/license — the repo already has
  them):
  ```bash
  gh repo create sundial --public --source . --remote origin --disable-wiki
  # or: create it in the web UI, then:
  # git remote add origin git@github.com:<you>/sundial.git
  ```
- [ ] Push `main` and confirm the default branch:
  ```bash
  git push -u origin main
  ```

## 2. Watch the FIRST CI run

`.github/workflows/ci.yml` has **never executed on a GitHub runner** — it's
only ever been proven via `scripts/verify_clean_checkout.sh` locally. Expect
the first run to need a fixup or two.

- [ ] Watch it:
  ```bash
  gh run watch
  ```
- [ ] Things most likely to bite on the Ubuntu runner (none are logic bugs,
  just environment):
  - GPU tests are `#[ignore]`d and must stay ignored on CI — the runner has no
    GPU. The CPU suite (`cargo test --workspace`) is what runs. Confirm the
    ignored GPU tests are actually skipped, not attempted.
  - `wasm-pack` install/version and the `wasm32-unknown-unknown` target must be
    provisioned in the workflow (the local machine had them; a clean runner may
    not). Fix the workflow, don't fix the code.
  - macOS-only assumptions (paths, toolchain) — the local dev machine is an M4
    Pro; CI is Linux.
- [ ] Iterate until the badge is green. Add a CI status badge to `README.md`
  once it passes.

## 3. Deploy the browser demo (you need its URL for step 5)

The demo is the whole pitch — get it live before anything else user-facing.
It's a static Vite build in `web/dist` (two pages: `index.html` transport hero,
`bench.html` drop-a-file). WebGPU requires HTTPS, which both options below give
you.

- [ ] Build the demo from a clean checkout. Per the M2 plan, the wasm bindings
  live in `crates/sundial-web` (package `sundial-lp`) after the final task; the
  web app imports the generated `pkg`. Build in this order (wasm first, then the
  web bundle):
  ```bash
  wasm-pack build crates/sundial-web --target web    # regenerates crates/sundial-web/pkg (gitignored)
  cd web && npm ci && npm run build                  # emits web/dist (index.html + bench.html + assets)
  cd ..
  ```
  (If a step can't find the package, confirm the final npm task rewired `web/`
  to import `sundial-web`'s `pkg` — the import path is the one thing to
  sanity-check against the actual built output.)
- [ ] **Option A — GitHub Pages (simplest).** Add a Pages deploy workflow (or
  use the Pages UI → "GitHub Actions") that runs the two build commands above
  and publishes `web/dist`. Then enable Pages in repo Settings → Pages. Your
  URL will be `https://<you>.github.io/sundial/`.
  - Vite serves from `/` by default; for a project Pages path
    (`/sundial/`) set `base: '/sundial/'` in `web/vite.config.*` (or deploy to a
    user/org root or a custom domain and skip this).
- [ ] **Option B — Netlify/Vercel.** Point it at the repo with build command
  `wasm-pack build crates/sundial-web --target web && cd web && npm ci && npm run build`
  and publish directory `web/dist`. Gives you a clean apex/custom domain.
- [ ] Open the deployed URL on a real machine and confirm end-to-end **before
  posting**: 32×32 reaches `Optimal (CPU f64 verified)`, the arriving-mass panel
  converges to the target, and the bench page solves a dropped `.mps`. Try it in
  Chrome and Safari (the two most common HN visitor browsers) — WebGPU floor is
  Chrome/Edge 113+, Firefox 141+, Safari 26+.
- [ ] Record the final URL — this is `<DEMO_URL>`.

## 4. Publish the npm package (`sundial-lp`)

- [ ] **Re-check the name is still free** (it was intended free at build time,
  but check again at publish):
  ```bash
  npm view sundial-lp
  ```
  - `npm view` errors with 404 → name is free, proceed.
  - If it's **taken**, fall back to the scoped name `@sundial/solver` (update
    `crates/sundial-web/Cargo.toml`'s package metadata / the generated
    `pkg/package.json` `name`, and the `npm install` line in
    `docs/writeup.md` §9 and `README.md`). If **both** are taken, stop and pick
    a new name before publishing.
- [ ] `npm login` (you'll need an npm account with publish rights).
- [ ] Build the publishable package and publish from the generated `pkg`:
  ```bash
  wasm-pack build crates/sundial-web --target web    # or --target bundler for bundler consumers
  cd crates/sundial-web/pkg
  # confirm package.json: name (sundial-lp or fallback), version, license,
  # repository URL, and that the files list includes the .wasm + .js + .d.ts
  npm publish --access public                        # --access public is required for a @scoped fallback
  cd ../../..
  ```
- [ ] Verify the published package installs and its snippet runs:
  ```bash
  cd $(mktemp -d) && npm install sundial-lp   # (or the fallback name)
  ```
  Confirm the `import init, { solveMps } from "sundial-lp"` snippet in the
  writeup matches the actual exported API (init function + `solveMps`).

## 5. Fill the demo URL into the writeup

- [ ] Replace **every** `<DEMO_URL>` in `docs/writeup.md` with the real URL from
  step 3 (the "Live demo" header line near the top and the §9 "Try it" block —
  grep to be sure you got them all):
  ```bash
  grep -n '<DEMO_URL>' docs/writeup.md   # find them all first
  ```
- [ ] Re-read §7's benchmark split and confirm it still matches the shipped
  `report.md` (regenerated during M2 Task 11).
- [ ] Commit the resolved writeup:
  ```bash
  git add docs/writeup.md && git commit -m "docs: resolve demo URL in launch writeup"
  git push
  ```

## 6. Post the Show HN

- [ ] Title (lead with the number; keep the technique/vendor jargon out of the
  title — HN rewards the story, punishes "linear programming" / "PDHG" in the
  headline). Options, best first:
  1. **Show HN: I solved a 1,048,576-variable optimization problem in a browser tab, on any GPU**
  2. Show HN: A GPU optimization solver in WebGPU — no CUDA, no install, no server
  3. Show HN: Sundial — watch a million-variable optimization problem solve on your own GPU
- [ ] First comment (post it yourself, immediately): the honest framing that
  makes the project credible to a skeptical audience — f32 GPU iterates but
  every "Optimal" is re-verified in f64 on the CPU (the GPU never grades its own
  homework); the 1e-4 default tier; the df64 negative result and *why* (Metal
  fast-math, traced to the compiler); simplex still wins on small LPs. Link the
  writeup (`docs/writeup.md`) and the repo. Pre-empting the skepticism is what
  lands this crowd.
- [ ] Have the repo, the live demo, and `npm install sundial-lp` all working
  *before* you post — the top comment will be someone trying all three.
- [ ] Post in the morning US time (Pacific) on a weekday for the widest window.

---

### Rollback / if something's wrong after posting

- npm: `npm deprecate sundial-lp@<version> "message"` (you can't un-publish
  after 72h; deprecate instead). Publish a patched `x.y.z+1`.
- Demo: the deploy is static — revert the offending commit and redeploy.
- Repo: it's public now; a force-push to rewrite history after clones exist is
  worse than a follow-up commit. Fix forward.
