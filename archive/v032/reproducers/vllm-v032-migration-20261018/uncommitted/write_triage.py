import collections, hashlib, pathlib, re, subprocess
out=pathlib.Path('/opt/project/evidence/vllm-v032-migration-20261018/uncommitted')
active='9a3af353e8'; upstream='5df63a7203cdb9328d824a27b682e77c7b9928be'
notes={
'csrc/libtorch_stable/moe/grouped_topk_kernels.cu':('Exact file integrated (SHA256 match). Preserve integrated native correction, no unique dirty delta.','Native file relocated to csrc/moe/grouped_topk_kernels.cu; old tanh sigmoid, epsilon normalization and subtract-bias remain. Integrated correction still required.'),
'tests/kernels/moe/test_grouped_topk.py':('Exact file integrated (SHA256 match).','Associated new numeric regression cases are absent upstream; retain integrated tests with native repair.'),
'tests/models/glm5next/test_kda_recurrent.py':('Exact file integrated (SHA256 match), including corrected logical-view padding case and prior accepted-state test.','New padding/prior-accepted-state regression cases absent in pinned upstream test file. Retain integrated tests.'),
'tests/quantization/test_exl3_prefill_plan.py':('Integrated superset: all added numerical-oracle/regression logic retained; active additionally covers int32/int64 and router dtype. Dirty copy loses that coverage.','File absent; custom EXL3 tests remain necessary. No extra dirty changes to port.'),
'tests/entrypoints/openai/responses/test_message_phases.py':('Lifecycle test integrated, GLM xfail removed; active additionally has substantial stream/tool lifecycle coverage. Dirty xfail is obsolete, not a missing fix.','Custom test file absent; upstream is not proof of these lifecycle contracts. Preserve active tests, not old xfail.'),
'tests/entrypoints/openai/responses/test_parsable_context_unit.py':('Tests retained and expanded in active: malformed JSON covers python in addition to search/container; additional active lifecycle tests missing from dirty copy.','Custom test file absent; corresponding new call-id preservation is upstream, but JSON retry response-item fix is not.'),
'tests/entrypoints/openai/responses/test_responses_utils.py':('New continuation/refusal tests integrated; dirty copy otherwise loses later rich-output, ownership and reasoning-preservation cases.','Upstream still drops previous function calls and reads only first response text part; retain active fix/tests.'),
'tests/parser/engine/replay_harness.py':('Exact file integrated (SHA256 match).','Local artifact tokenizer fixture helper absent upstream; retain active test fixture.'),
'tests/parser/engine/test_qwen3.py':('Tests integrated/expanded: MiMo-only legacy JSON and required JSON cases generalized to profile={GLM,MiMo}; obsolete GLM xfails removed; active adds further contracts.','Custom selected-parser/reentry/JSON regression cases absent upstream. Retain active versions.'),
'tests/v1/spec_decode/test_dflash_model_policy.py':('Added policy/padding tests integrated; active additionally supplies explicit model_type in padding calls and broader subsequent contracts.','File absent; model-specific DFlash policy is local and remains migration work.'),
'vllm/entrypoints/openai/responses/context.py':('Added typed call-id preservation and JSON retry response item retained; active extends parser lifecycle, pending parallel dispatch, python retry, rich output. No unique dirty repair.','Typed ResponseFunctionToolCall and call_id=last_msg.call_id already upstream. _create_parsable_json_parse_error_messages absent: search/container still call chat-message error helper. Do not replay the whole older file.'),
'vllm/entrypoints/openai/responses/utils.py':('Previous-response function-call continuation and multipart output/refusal concatenation integrated; active adds history ownership and rich output handling.','Upstream still loops only ResponseOutputMessage from previous response and reads item.content[0].text; retain integrated repair.'),
'vllm/model_executor/models/dflash_policy.py':('Exact file integrated (SHA256 match).','File absent; local MiMo-specific parallel-padding policy remains migration work.'),
'vllm/parser/mimo.py':('Serving adapter, reentry, initial state and JSON contracts integrated in evolved implementation. Old inline JSON scanner superseded by shared json_tools._RequiredJSONStream; active adds token-ID alignment/accounting and parse_deltas. Dirty copy would regress those.','No MiMoServingParser or local reentry/JSON serving path; retain active implementation adapted to upstream contracts, not old cumulative scanner.'),
'vllm/parser/parser_manager.py':('MiMo selection block present in active; active additionally selects GLM serving adapter. No unique dirty repair.','No MiMoServingParser selection in pinned upstream; retain integrated selection when adapting parser.'),
}
rows=[]
for lane in ('numerics','parser-padding'):
 r=pathlib.Path('/opt/project/builds/glm-mimo-sol-'+lane+'-20261017')
 diff=(out/(lane+'.hunks.patch')).read_text(); file=None; n=0
 for line in diff.splitlines():
  if line.startswith('diff --git '):file=line.split(' b/',1)[1]
  if line.startswith('@@'):
   n+=1; a,u=notes[file]; rows.append((lane,n,file,line,a,u))
 for f in subprocess.check_output(['git','-C',str(r),'diff','--name-only','HEAD'],text=True).splitlines():
  mapped='csrc/moe/grouped_topk_kernels.cu' if f=='csrc/libtorch_stable/moe/grouped_topk_kernels.cu' else f
  result=subprocess.run(['git','-C',str(r),'show',upstream+':'+mapped],capture_output=True)
  dest=out/(lane+'.v032.'+f.replace('/','__')+'.txt')
  dest.write_bytes(result.stdout if result.returncode==0 else b'ABSENT IN PINNED UPSTREAM\n')
lines=['# Dirty worktree hunk dispositions','',f'Comparison pins: integrated `{active}`; upstream `{upstream}`. Original HEAD for both trees: `0cd0073dffb2158360ca44114031a9ab55ce7034`.','',f'Coverage: {len(rows)}/{len(rows)} original unified-zero hunks across 15 modified files (4 numerics, 11 parser-padding). No untracked files in captured porcelain status. This is bounded delta triage, not runtime review or test acceptance.','', '| Lane/hunk | Original hunk / file | Integrated disposition | Pinned .32 disposition |','|---|---|---|---|']
for lane,n,f,h,a,u in rows:lines.append(f'| {lane}/{n} | `{f}` `{h}` | {a} | {u} |')
lines += ['','## Cleanup conclusion','','Both dirty worktrees can be removed after retaining this evidence and confirming the source owner retains integrated `9a3af353e8` or its intended replacement. There are **zero useful changes unique to these dirty worktrees**. No source-owner cherry-pick from them is required. Do not overwrite integrated files with these older working copies.','', 'The staged original patches preserve exact recoverable dirty deltas (`*.diff.patch`), and `*.vs-integrated.patch` records why whole-file copies are unsafe. The original hunk ledger uses original diff boundaries, including all import-only and call-site template hunks.','', 'No obsolete synthetic-phase implementation is introduced by these uncommitted patches: `test_message_phases.py` is a filename, not evidence that every test is synthetic. Its added lifecycle test is useful and already integrated; only its old GLM xfail is obsolete.','', 'No source edits, deletion, branch creation, builds, GPU activity, tests, service actions or Obsidian writes performed. Runtime equivalence of migrated implementations remains outside this triage.']
(out/'DISPOSITION.md').write_text('\n'.join(lines)+'\n')
print(collections.Counter(row[0] for row in rows));print('total hunks',len(rows))
