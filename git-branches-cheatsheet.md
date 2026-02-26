# Git Branches Cheat Sheet

A quick reference for everyday branch operations.

---

## Seeing What's Going On

```bash
git branch              # List local branches (* = current)
git branch -r           # List remote branches
git branch -a           # List all (local + remote)
git status              # Show current branch + uncommitted changes
git log --oneline --graph --all   # Visual tree of all branches
```

---

## Creating Branches

```bash
# Create a new branch (stays on current branch)
git branch my-feature

# Create AND switch to it immediately  (most common)
git checkout -b my-feature

# Modern equivalent (Git 2.23+)
git switch -c my-feature

# Create a branch from a specific branch or commit
git checkout -b my-feature main
git checkout -b my-feature abc1234
```

---

## Switching Branches

```bash
git checkout my-feature       # Switch to branch
git switch my-feature         # Modern equivalent (Git 2.23+)
git checkout -                # Switch back to previous branch (handy!)
```

---

## Merging into Master / Main

```bash
# Step 1 — Go to the target branch (master or main)
git checkout main

# Step 2 — Merge your feature branch in
git merge my-feature

# Merge with a commit message even if fast-forward is possible
git merge --no-ff my-feature

# Abort a merge gone wrong
git merge --abort
```

---

## Rebasing (alternative to merge, cleaner history)

```bash
# Go to your feature branch
git checkout my-feature

# Rebase on top of main
git rebase main

# Interactive rebase (squash, reorder, edit commits)
git rebase -i main

# Abort a rebase gone wrong
git rebase --abort
```

---

## Deleting Branches

```bash
# Delete a local branch (safe — only if fully merged)
git branch -d my-feature

# Force delete a local branch (even if NOT merged) ⚠️
git branch -D my-feature

# Delete a remote branch
git push origin --delete my-feature

# Prune stale remote-tracking refs (clean up after remote deletes)
git fetch --prune
```

---

## Pushing & Tracking Remote Branches

```bash
# Push branch to remote and set upstream tracking
git push -u origin my-feature

# After -u is set once, you can just use:
git push
git pull

# Pull a remote branch you don't have locally
git checkout -b my-feature origin/my-feature

# Modern equivalent
git switch --track origin/my-feature
```

---

## Renaming a Branch

```bash
# Rename current branch
git branch -m new-name

# Rename a specific branch
git branch -m old-name new-name

# Push renamed branch and reset upstream
git push origin -u new-name
git push origin --delete old-name
```

---

## Stashing (when you need to switch but have uncommitted work)

```bash
git stash               # Save uncommitted changes temporarily
git stash pop           # Restore them on current branch
git stash list          # See all stashes
git stash drop          # Delete the latest stash
```

---

## Comparing Branches

```bash
# Show commits in my-feature not yet in main
git log main..my-feature

# Show diff between two branches
git diff main my-feature

# Show which branches are already merged into main
git branch --merged main

# Show which branches are NOT yet merged
git branch --no-merged main
```

---

## Typical Full Workflow (feature branch → main)

```bash
git checkout main                  # Start from main
git pull                           # Make sure it's up to date
git checkout -b my-feature         # Create and switch to new branch

# ... do your work, then:
git add .
git commit -m "feat: my changes"

git checkout main                  # Go back to main
git pull                           # Pull any new changes
git merge --no-ff my-feature       # Merge feature in
git push                           # Push to remote

git branch -d my-feature           # Clean up local branch
git push origin --delete my-feature  # Clean up remote branch
```

---

> **Tip:** If you're ever unsure what branch you're on, just run `git status` — it always tells you at the top.
