# RELEASE.md — Sundial publish checklist

This is the human-run checklist to take Sundial from a local, never-published
repo to a live Show HN. Work top to bottom; each step is a checkbox. Commands
assume you are in the repo root on the branch you intend to ship.

**Progress as of 2026-08-08:** §0 is complete except the two decisions that are
yours to make (visibility, code of conduct) — the ship branch is merged to `main`
and `scripts/verify_clean_checkout.sh` reports ALL GREEN. §1 (repo created and
pushed) and §2 (first CI run green) are done — the remote is
`https://github.com/NMitchem/Sundial`. The repo is still **private**; flipping it
to public is the first thing left. Everything from §3 (demo deploy) onward is
unexecuted.

Order matters: get the repo public and CI green first, then the demo deploy
(you need its URL), then npm, then fill the URL into the writeup, then post.

---

## 0. Decisions to make before you touch anything

- [x] **Repo name.** Decided: **`NMitchem/Sundial`**. (The local working directory
  is still `or-fable`; the crates and the npm package are `sundial-*` /
  `sundial-lp`. `Cargo.toml`'s `repository` field matches the repo, capital S
  included.)
- [ ] **Visibility.** Public. (It's a Show HN.) **Still private — do this:**
  ```bash
  gh repo edit NMitchem/Sundial --visibility public
  ```
- [x] **License is already MIT OR Apache-2.0** — `LICENSE-MIT` and
  `LICENSE-APACHE` confirmed present at the root, matching `Cargo.toml`.
  `NOTICE` records the provenance of the bundled Netlib fixtures and TLC taxi
  extract.
- [ ] **Code of conduct — add one when there's a community to govern.** There is
  deliberately no `CODE_OF_CONDUCT.md`: with a single maintainer and no
  contributors it would document a process that doesn't exist. Add the
  Contributor Covenant with a real reporting address when you enable Discussions
  or merge a first outside PR.
- [x] **Merge to `main`** — DONE (2026-08-08). `oss-prep` fast-forwarded into
  `main`, and `scripts/verify_clean_checkout.sh` (fmt + workspace tests + wasm +
  web build, from a scratch clone) reported **CLEAN CHECKOUT: ALL GREEN**. Re-run
  it if you touch anything before publishing:
  ```bash
  bash scripts/verify_clean_checkout.sh
  ```
- [ ] **Git history still contains the planning scaffolding.** `30afa64` removed
  `docs/superpowers/` and `or-project-proposals.md` from the tree, but their full
  text stays recoverable with `git show` on the ~30 commits that carried them.
  There are no secrets in them — it is internal planning prose. Decide before the
  repo goes public: leave it (normal for an unsquashed history), or rewrite now,
  which is still cheap because `main` is the only pushed branch and the repo is
  private. This is the last irreversible choice on the list.

## 1. Create the GitHub repo and push — DONE (2026-07-19)

- [x] Repo created as `NMitchem/Sundial` (private at creation).
- [x] `main` pushed and set as the default branch.

## 2. Watch the FIRST CI run — DONE (2026-07-19)

- [x] `.github/workflows/ci.yml` ran on a GitHub runner and **passed on the first
  attempt** (ubuntu-latest, 3m20s) — none of the environment fixups anticipated
  below were needed.
- [ ] Add a CI status badge to `README.md`. (Deferred until the repo is public —
  the badge image 404s on a private repo.)
  (For the record, the things expected to bite and didn't: GPU tests staying
  `#[ignore]`d on a GPU-less runner, `wasm-pack` / `wasm32-unknown-unknown`
  provisioning, and macOS-only toolchain assumptions. If CI ever breaks on those,
  fix the workflow, not the code.)

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
  bash scripts/build_npm.sh                          # release wasm build into crates/sundial-web/pkg (gitignored)
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
  `bash scripts/build_npm.sh && cd web && npm ci && npm run build`
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
  bash scripts/build_npm.sh                          # release wasm build, then copies types-extra.d.ts and
                                                      # adds it + both LICENSE files to package.json's "files"
                                                      # allowlist — publishing via a bare `wasm-pack build` skips
                                                      # this step and ships a tarball missing the .d.ts and both
                                                      # LICENSE files; check the "== pack dry run" output it prints
                                                      # for the .wasm + .js + .d.ts + LICENSE-* file list
  cd crates/sundial-web/pkg
  # confirm package.json: name (sundial-lp or fallback), version, license, repository URL
  npm publish --access public                        # --access public is required for a @scoped fallback
  cd ../../..
  ```
- [ ] Verify the published package installs and its snippet runs:
  ```bash
  cd $(mktemp -d) && npm install sundial-lp   # (or the fallback name)
  ```
  Confirm the `import init, { solveMps } from "sundial-lp"` snippet in the
  writeup matches the actual exported API (init function + `solveMps`).

## 4b. Publish the Rust crates (crates.io)

This is what makes the project findable by Rust users at all — crates.io and
docs.rs are where they look, and docs.rs pages rank in search. All four crates
carry `description`, `keywords`, `categories`, `readme`, and `repository`.

**Order matters — each crate must exist on the index before its dependents can
be published.** Until `sundial-core` is up, a dry run of the others fails with
`no matching package named 'sundial-core' found`; that is expected, not a
manifest bug.

```bash
cargo publish -p sundial-core     # no path deps; publish first
cargo publish -p sundial-mps      # depends on sundial-core
cargo publish -p sundial-cli      # depends on both
cargo publish -p sundial-lp       # optional; the npm package is its real home
```

- [ ] Dry-run `sundial-core` first and read the file list:
  ```bash
  cargo publish -p sundial-core --dry-run
  cargo package -p sundial-core --list | head -40
  ```
- [ ] Publish in the order above, waiting for each to appear on the index
  before the next.
- [ ] Confirm docs.rs built each crate (it can take a few minutes) and that the
  landing page shows the README rather than a bare module list.
- [ ] `cargo install sundial-cli` in a clean environment and run
  `sundial solve` on a fixture.

Names are unclaimed as of this writing — verify with `cargo search sundial-core`
before you count on them.

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
