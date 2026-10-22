from pathlib import Path
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017');p=r/'build/build_image.sh';s=p.read_text()
a='''base_tag="vllm-v${VLLM_VERSION}-source:${source_commit:0:12}"
python3 "$here/provenance.py"'''
b='''# Docker bind-mounts .git while deriving SCM versions. A linked worktree's
# .git file points outside its context, so give Docker a standalone clone.
if [[ -f "$repo/.git" ]]; then
  context_parent="$repo/.build-contexts"
  mkdir -p "$context_parent"
  context_repo=$(mktemp -d "$context_parent/${source_commit:0:12}.XXXXXX")
  branch=$(git -C "$repo" symbolic-ref --quiet --short HEAD) || {
    echo "linked source worktree must have a branch for context staging" >&2
    exit 1
  }
  git clone --no-local --no-hardlinks --single-branch --no-tags \\
    --branch "$branch" --no-checkout "$repo" "$context_repo"
  git -C "$context_repo" fetch --no-tags "$repo" \\
    "refs/tags/v${VLLM_VERSION}:refs/tags/v${VLLM_VERSION}"
  git -C "$context_repo" checkout --detach "$source_commit"
  test "$(git -C "$context_repo" rev-parse HEAD)" = "$source_commit"
  test -d "$context_repo/.git"
  test ! -f "$context_repo/.git/objects/info/alternates"
  repo=$context_repo
  here="$repo/build"
fi

base_tag="vllm-v${VLLM_VERSION}-source:${source_commit:0:12}"
python3 "$here/provenance.py"'''
assert a in s;s=s.replace(a,b);p.write_text(s)
p=r/'.gitignore';s=p.read_text();s+='\n# Standalone Docker contexts for linked release worktrees.\n/.build-contexts/\n';p.write_text(s)
p=r/'build/README.md';s=p.read_text().replace('The three references above are the verified Linux ARM64 child manifests\nresolved from Docker Hub on 2026-10-08; the build records them unchanged in\nits provenance file.','The three immutable references above were read back from Docker Hub on\n2026-10-18, including their Linux ARM64 config metadata. This verifies\nidentity and platform, not a new native build or runtime qualification.')
s+='''\n## Linked worktree staging\n\nThe build script stages a standalone clone beneath `.build-contexts/` when\nthe release checkout is a linked worktree. Docker receives an internal `.git`\ndirectory, the verified official version tag, and no object alternates. The\nsource commit is unchanged; version metadata is not synthesized. Context\ncreation and native builds remain deferred until the build resource envelope\nand any required maintenance window are established.\n''';p.write_text(s)
