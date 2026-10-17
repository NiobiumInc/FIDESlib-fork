#!/usr/bin/env bash
# Reports how far this fork has drifted from the upstream project: the commits on
# each side since the merge base, and what a merge of upstream into the current
# commit would conflict on, counted twice (as is, and ignoring whitespace). The
# merge is only tried: it is aborted, nothing is committed or pushed.
#
# Writes a Markdown report to $GITHUB_STEP_SUMMARY (or stdout) and the same numbers
# as JSON to <out dir>/upstream-sync.json. Always exits 0 unless git itself fails:
# this is a report, not a check.
#
# Usage: upstream_sync_check.sh <out dir> [upstream url] [upstream branch]
set -uo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <out dir> [upstream url] [upstream branch]" >&2
  exit 2
fi
out=$1
url=${2:-https://github.com/CAPS-UMU/FIDESlib.git}
branch=${3:-main}
summary=${GITHUB_STEP_SUMMARY:-/dev/stdout}
mkdir -p "$out" || exit 1

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "the work tree has uncommitted changes; run this on a clean checkout" >&2
  exit 1
fi

git fetch --quiet --no-tags "$url" "refs/heads/$branch" || exit 1
upstream=$(git rev-parse FETCH_HEAD) || exit 1
ours=$(git rev-parse HEAD) || exit 1
base=$(git merge-base "$ours" "$upstream") || exit 1
upstream_ahead=$(git rev-list --count "$ours..$upstream")
ours_ahead=$(git rev-list --count "$upstream..$ours")

# Tries the merge with the given extra arguments, prints "<files> <lines>" for the
# conflicts, and the per-file line counts to the file named by $2.
try_merge() {
  local args=$1 per_file=$2 files lines
  git -c user.name=upstream-sync -c user.email=upstream-sync@invalid \
    merge --no-commit --no-ff $args "$upstream" > /dev/null 2>&1
  : > "$per_file"
  while IFS= read -r f; do
    if [ -f "$f" ]; then
      n=$(awk '/^<<<<<<< /{c=1} c{n++} /^>>>>>>> /{c=0} END{print n+0}' "$f")
    else
      n=0  # deleted on one side, or a submodule
    fi
    printf '%s\t%s\n' "$n" "$f" >> "$per_file"
  done < <(git diff --name-only --diff-filter=U)
  git merge --abort > /dev/null 2>&1 || git reset --quiet --merge
  files=$(wc -l < "$per_file")
  lines=$(awk -F'\t' '{s+=$1} END{print s+0}' "$per_file")
  echo "$files $lines"
}

read -r plain_files plain_lines < <(try_merge "" "$out/conflicts.tsv")
read -r ws_files ws_lines < <(try_merge "-Xignore-all-space" "$out/conflicts-ignore-whitespace.tsv")

{
  echo "### Drift from upstream"
  echo
  echo "| Item | Value |"
  echo "|---|---|"
  echo "| Upstream | \`$url\` branch \`$branch\` at \`${upstream:0:8}\` |"
  echo "| This commit | \`${ours:0:8}\` |"
  echo "| Merge base | \`${base:0:8}\` ($(git log -1 --format=%cs "$base")) |"
  echo "| Upstream commits not here | $upstream_ahead |"
  echo "| Commits here not upstream | $ours_ahead |"
  echo
  echo "### Trial merge of upstream into this commit"
  echo
  echo "| Merge | Conflicted files | Conflict lines |"
  echo "|---|---|---|"
  echo "| As is | $plain_files | $plain_lines |"
  echo "| Ignoring whitespace | $ws_files | $ws_lines |"
  echo
  if [ "$ws_files" -gt 0 ]; then
    echo "### Largest conflicts, ignoring whitespace"
    echo
    echo "| File | Conflict lines |"
    echo "|---|---|"
    sort -t$'\t' -k1,1nr "$out/conflicts-ignore-whitespace.tsv" | head -15 |
      awk -F'\t' '{printf "| `%s` | %s |\n", $2, ($1 == 0 ? "deleted on one side" : $1)}'
    echo
  fi
  if [ "$upstream_ahead" -gt 0 ]; then
    echo "### Newest upstream commits not here"
    echo
    echo "| Commit | Date | Subject |"
    echo "|---|---|---|"
    git log --no-merges -20 --format='%h%x09%cs%x09%s' "$ours..$upstream" |
      awk -F'\t' '{gsub(/\|/, "\\|", $3); printf "| `%s` | %s | %s |\n", $1, $2, $3}'
    echo
  fi
} >> "$summary"

cat > "$out/upstream-sync.json" <<EOF
{
  "upstream_url": "$url",
  "upstream_branch": "$branch",
  "upstream_commit": "$upstream",
  "commit": "$ours",
  "merge_base": "$base",
  "upstream_commits_not_here": $upstream_ahead,
  "commits_here_not_upstream": $ours_ahead,
  "conflicted_files": $plain_files,
  "conflict_lines": $plain_lines,
  "conflicted_files_ignoring_whitespace": $ws_files,
  "conflict_lines_ignoring_whitespace": $ws_lines
}
EOF
exit 0
