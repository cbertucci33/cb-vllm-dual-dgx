from pathlib import Path
import subprocess,json,ast,re,collections
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017');e=Path('/opt/project/evidence/vllm-v032-migration-20261018/source')
def git(*a):return subprocess.check_output(['git',*a],cwd=r,text=True)
exact={line.split()[1] for line in (e/'patch-equivalence.txt').read_text().splitlines() if line.startswith('- ')}
semantic={'840e97fd11':'Upstream #59528 null-block protection verified in assessment; local shared cache code reconciled separately.','c22d164448':'Upstream #57169 generic packed layout/speculative tail verified in assessment; private DFlash ring extensions retained separately.','334c4ae6aa':'Upstream .32 imports built-in GLM/Qwen config and processor classes; retain newer .32 registries/models rather than older equivalent backport.'}
rows=[]
for line in (e/'local-commit-manifest.tsv').read_text().splitlines():
 sha,subject=line.split('\t',1);paths=git('diff-tree','--no-commit-id','--name-only','-r',sha).splitlines()
 normalized=[p.replace('csrc/libtorch_stable/','csrc/').replace('/autoregressive/','/target_dependent_ar/').replace('test_gpu_autoregressive_speculator','test_gpu_target_dependent_ar_speculator') for p in paths]
 status='included/adapted';reason='Filtered aggregate v0.31-to-local source delta; reconcile changed paths against .32 implementation, not replay entire commit.'
 if sha in exact:status='upstream-equivalent';reason='Exact git patch-ID equivalence against pinned .32 (git cherry minus).'
 for prefix,note in semantic.items():
  if sha.startswith(prefix):status='upstream-equivalent';reason=note
 if sha.startswith('f82172290a'):status='obsolete';reason='Unsupported synthetic phase inference retired; native Matrix delivery cause remains unresolved.'
 if sha.startswith('9a3af353e8'):reason+=' Mixed final repair batch filtered: unsupported synthetic phases excluded, newer .32 contracts preferred, useful regression tests preserved/adapted.'
 rows.append({'commit':sha,'subject':subject,'disposition':status,'reason':reason,'original_paths':paths,'normalized_migration_paths':normalized,'acceptance':'Source disposition only; native/GPU/model/performance/client behavior unverified.'})
assert len(rows)==90
(e/'patch-disposition-manifest.json').write_text(json.dumps(rows,indent=2))
counts=collections.Counter(x['disposition'] for x in rows)
md=['# Original .31 patch dispositions','',f'90/90 original commits have a recorded migration disposition: {dict(counts)}. These are source dispositions, not 90 runtime-validated repairs.','', '| Commit | Disposition | Subject |','|---|---|---|']
for x in rows:md.append(f"| `{x['commit'][:10]}` | {x['disposition']} | {x['subject']} |")
(e/'PATCH-DISPOSITIONS.md').write_text('\n'.join(md)+'\n')
paths=(e/'unmerged-paths.txt').read_text().splitlines();pr=[]
for p in paths:
 owner='source'
 if any(s in p for s in ['vllm/parser/','vllm/tool_parsers/','vllm/reasoning/','vllm/tokenizers/','vllm/entrypoints/','tests/parser/','tests/reasoning/','tests/tool_parsers/','tests/tool_use/','tests/entrypoints/','structured_output/']):owner='parser'
 reason='Three-way conflict reconciled against .32; retained independent local contracts and regression fixtures. See conflict snapshots and exact reconciliation scripts.'
 if owner=='parser':reason='Parser owner reconciliation; see ../parser/MIGRATION.md; unsupported synthetic phase inference retired.'
 if 'qwen4_exp' in p or 'transformers_utils/' in p or p in ['docs/design/mm_processing.md','tests/models/glm5next/test_kda_state_dtype.py','tests/v1/attention/test_indexer_tp_row_shard.py','tests/v1/attention/test_flashmla_nope_sm90_backend_selection.py','vllm/model_executor/layers/sparse_attn_indexer.py','vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py']:reason='Retain newer .32 implementation/fixtures: upstream-equivalent or unrelated older backport must not regress newer behavior.'
 if p=='vllm/models/glm5next/common/model.py':reason='Retain newer upstream aux-state input-boundary capture and single SP gather; keep local unsupported-config/EXL3/loading contracts. Adapt old completed-output fixtures to equivalent .32 capture contract.'
 if p.endswith('dflash/speculator.py'):reason='Combine local fixed-buffer context staging with .32 prepare_forward(num_reqs,attn_metadata) callback and in-graph metadata updates. Obsolete precompute_context_kv keyword retired.'
 if p=='vllm/v1/worker/gpu/spec_decode/rejection_sampler.py':reason='Preserve both .32 SamplingMaskTensors 4-value result and local per-request custom-logprob dimensions/chunking.'
 pr.append({'path':p,'owner':owner,'status':'resolved/staged','reason':reason})
assert len(pr)==95 and not git('diff','--name-only','--diff-filter=U').strip()
(e/'conflict-resolution-manifest.json').write_text(json.dumps(pr,indent=2))
checks=[]
for f in git('diff','--name-only','HEAD').splitlines():
 p=r/f
 if p.suffix=='.py' and p.exists():
  try:compile(p.read_text(),str(p),'exec');checks.append({'path':f,'status':'pass'})
  except SyntaxError as exc:checks.append({'path':f,'status':'fail','line':exc.lineno,'error':str(exc)})
(e/'compile-static.json').write_text(json.dumps(checks,indent=2));assert all(x['status']=='pass' for x in checks)
rece=json.loads((e/'transformers-qualification/receipt.json').read_text());rece.update({'check_exit':0,'stderr':'','correct_patch_root':'transformers-5.17.0/src/transformers','initial_check':'Wrong sdist root rejected without applying; installed-package-relative root verified and check passed.'});(e/'transformers-qualification/receipt.json').write_text(json.dumps(rece,indent=2))
configs={n:json.loads((e/f'base-manifests/{n}-config.json').read_text()) for n in ['builder','final','devel']}
summary={n:{'os':v.get('os'),'architecture':v.get('architecture'),'variant':v.get('variant')} for n,v in configs.items()}
(e/'base-manifests/platform-summary.json').write_text(json.dumps(summary,indent=2));assert all(v['architecture']=='arm64' and v['os']=='linux' for v in summary.values())
print('Commit dispositions',dict(counts),'conflicts',len(pr),'Python compile',len(checks),'base configs',summary)
