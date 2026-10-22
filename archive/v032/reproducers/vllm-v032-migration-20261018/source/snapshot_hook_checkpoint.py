from pathlib import Path
import subprocess,json,hashlib,ast,sqlite3
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017');e=Path('/opt/project/evidence/vllm-v032-migration-20261018/source')
def git(*a):return subprocess.check_output(['git',*a],cwd=r)
assert not git('diff','--name-only').strip() and not git('diff','--name-only','--diff-filter=U').strip()
tree=git('write-tree').decode().strip();head=git('rev-parse','HEAD').decode().strip();names=git('diff','--cached','--name-only').decode().splitlines();rows=[];comp=[]
for f in names:
 p=r/f
 if not p.exists():rows.append({'path':f,'status':'deleted'});continue
 data=p.read_bytes();rows.append({'path':f,'sha256':hashlib.sha256(data).hexdigest(),'git_blob':git('rev-parse',f':{f}').decode().strip()})
 if p.suffix=='.py':compile(data,str(p),'exec');comp.append(f)
patch=git('diff','--cached','--binary');(e/'final-staged-binary.patch').write_bytes(patch);(e/'final-staged-source-hashes.json').write_text(json.dumps(rows,indent=2))
passed=['ruff-check','ruff-format','typos','clang-format','markdownlint-cli2','shellcheck','check-test-tethering','check-spdx-header','check-root-lazy-imports','check-filenames','check-forbidden-imports','check-torch-cuda-call','validate-config','validate-docker-versions','check-boolean-context-manager']
blocked={
 'mypy':'Exact .32 environment additional dependency pydantic==2.13.5 differs from existing cached environment unpinnedpydantic; exact environment creation requires package installation not authorized in this lane.',
 'pip-compile':'CUDA test lock re-resolution not run; no package resolver/resource admission or lockfile mutation authorized.',
 'pip-compile-rocm':'common.txt triggers unrelated ROCm lock re-resolution; not run under scope constraint.',
 'pip-compile-xpu':'common.txt triggers unrelated XPU lock re-resolution; not run under scope constraint.',
 'pip-compile-cpu':'common.txt triggers CPU test lock re-resolution; no resolver/resource admission authorized.',
 'format-torch-nightly-test':'Would regenerate requirements/test/nightly-torch.txt from changed CUDA requirements; lockfile work deferred, not passed.',
 'update-dockerfile-graph':'Changed Dockerfile triggers docker run of dockerfilegraph image; container/resource operation not authorized in this lane.',
 'signoff-commit':'Commit-msg stage pending an actual authorized commit; no commit attempted.'}
not_applicable=['actionlint','check-json','check-label-rules','check-agent-files','check-buildkite-step-keys','png-lint','test-nonroot-entrypoint','rust-cargo-autoinherit','rust-cargo-sort','rust-cargo-fmt','pip-compile-docs']
metadata={
 'head':head,'branch':git('branch','--show-current').decode().strip(),'staged_tree':tree,'staged_paths':len(names),'unstaged_paths':0,'unmerged_paths':0,'compiled_without_import':len(comp),'staged_binary_patch_sha256':hashlib.sha256(patch).hexdigest(),'precommit_version':'4.6.2','installed_hooks':['pre-commit','commit-msg'],'hook_interpreter_owner':'/opt/project/builds/vllm-v031-glm-compact-kv-20261010/.venv','passed_cached_staged_hooks':passed,'blocked_hooks':blocked,'no_matching_changed_paths_not_run':not_applicable,'informational_not_run':['suggestion'],'full_precommit_pass':False,'no_hook_bypass':True,'no_package_install':True,'no_compiler_container_GPU_or_service_action':True,'commit_attempted':False}
(e/'HOOK-CHECKPOINT.json').write_text(json.dumps(metadata,indent=2));(e/'compile-static-final.json').write_text(json.dumps({'passed':comp,'count':len(comp),'method':'compile source only, no import/execution'},indent=2))
p=e/'BUILD-HANDOFF.md';s=p.read_text().replace('704b5ef9522a3e4f4431e139b749ffe5510a012a',tree).replace('270 staged paths',f'{len(names)} staged paths').replace('- Existing core.hooksPath and executable pre-commit hook absent; no hooks were bypassed or dependencies installed. Repository author identity not configured; parent will use its verified existing agent identity, not impersonate a human.','- Existing compatible pre-commit4.6.2 installed both required Git hooks. Their interpreter depends on the retained compact-kv20261010/.venv; preserve that tool owner. No hook bypass, package install, compiler/container operation or commit attempted.')
s+='\n## Required hooks checkpoint — not committable yet\n\n15 cached staged hooks passed: '+', '.join(passed)+'. All actual failures observed in those cached checks were repaired and rechecked. Changes were narrow: native/style formatting, README heading order, equivalent partial-token fixture expression, four automatic SPDX headers, documented exact-file bootstrap/interpreter exemptions using the checker’s existing allowed_files mechanism, one regex import covered by requirements/common.txt, three test calls adapted to source-verified Torch2.13 accelerator APIs, and generated docker/versions.json from the current Dockerfile. No AMD/XPU lockfiles changed.\n\nFull required pre-commit has NOT passed. Blocked/unrun gates:\n\n'
for h,why in blocked.items():s+=f'- `{h}`: {why}\n'
s+='\nNo-match hook IDs and final source/patch hashes are recorded in HOOK-CHECKPOINT.json and final-staged-source-hashes.json. final-staged-binary.patch is the resumable full tracked delta from pinned .32. Do not commit until the required blocked gates are handled without bypassing them.\n';p.write_text(s)
with (e/'ACTIONS.md').open('a') as f:f.write('\n- Installedrequiredprecommit/commit-msghooksusingexisting4.6.2withoutenvinstallation.15cachedstagedhookspass after narrowrepairs. Fullrequiredhooksetblocked; noSKIP/nohookbypass/nocommit. Finalstagedtree '+tree+'; exactbinarypatch/hashmanifestandblockedhookledgerretained.\n')
print(json.dumps(metadata,indent=2))
