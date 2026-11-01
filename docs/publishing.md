# Release from the isolated source branch

## Repository layout

The original development repository maintains a **history-isolated** Git branch:

`release/static-v0.1`

Its contents are also checked out in a dedicated worktree. This branch has **no shared ancestor** with the development `main`; it was bootstrapped from the sanitized public repository, not from the development history.

The separately initialized **public repository** is `aerialnav-static`, with `main` as its publication branch. It is the **only** repository that should be connected to the future public GitHub remote.

## Update workflow

For the first push, create an **empty** GitHub repository without an initial README, license or `.gitignore`. Authenticate on this machine using `gh auth login` (or your configured SSH key). Keep credentials out of chat. Supply the repository URL and explicitly authorize pushing the standalone repository's `main` branch.

1. Develop public changes on the `release/static-v0.1` worktree. Do not merge development `main` into the release branch. Review all imported changes and commit them with publishable attribution.
2. Run the public-source audit and targeted regression checks:

   ```bash
   python scripts/audit_release.py --weights
   python -m ruff check src scripts tests
   python -m compileall -q src scripts tests
   PYTHONPATH=src python -m pytest -q -m 'not integration' tests
   ```

3. In the **independent public repository**, fetch **only** the release branch from the local development repository:

   ```bash
   git fetch --no-tags /path/to/development/repository \
     refs/heads/release/static-v0.1
   git merge --ff-only FETCH_HEAD
   ```

4. Check the standalone public repository before considering a push:

   ```bash
   git status --short
   git log --oneline --all
   git rev-list --count main
   python scripts/audit_release.py
   ```

5. After reviewing the public content and recorded validation status, add the intended GitHub remote to the standalone repository and push `main` explicitly.

   ```bash
   git remote add origin <user-provided-github-repository-url>
   git push -u origin main
   ```

The placeholders above are not literal filesystem paths or URLs.

## Security constraints

- **Never** push the development repository using `--all`, `--mirror`, or `--tags`. Those operations can expose private branches, Git objects and historical metadata.
- Do not run `git merge main` on the orphan release branch. That would attach the private development ancestry to the public branch.
- Do not copy `.git` from the development repository into the standalone public repository.
- Keep the public branch history independent. An explicit one-branch fetch followed by `git merge --ff-only` is the supported update path.
- Upstream code attribution and third-party licensing notices must remain intact; old internal task or personally named identifiers must not be restored.
- Keep simulator smoke results, selected flight demos, training success rates and held-out evaluation rates distinct. The research baseline can be published with its recorded limitations; advertised held-out performance requires independent evaluation of the exported checkpoint.

## Status

The branch now carries the selected dynamic direct-navigation baseline. The user-authorized N4 simulator smoke passed: two training updates and four packaged-checkpoint inference episodes. A full held-out evaluation remains outstanding. Source, geometry, metadata and packaging checks are recorded in `docs/release_checklist.md`. No remote GitHub publication has been performed.
