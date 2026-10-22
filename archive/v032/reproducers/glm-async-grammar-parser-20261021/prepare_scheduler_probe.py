from pathlib import Path
import ast
d=Path('/opt/project/evidence/glm-async-grammar-parser-20261021')
src=Path('/opt/project/worktrees/glm32-three-improvements-20261020/vllm/v1/core/sched/scheduler.py').read_text()
cls=next(x for x in ast.parse(src).body if isinstance(x,ast.ClassDef) and x.name=='Scheduler')
methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ['update_draft_token_ids_in_output','get_grammar_bitmask']]
s=(d/'mask_probe.py').read_text().split('cases=0')[0]
s+="\nfrom vllm.v1.core.sched.output import GrammarOutput\nfrom vllm.v1.structured_output.utils import strip_speculative_padding\n"
s+='\n'.join(ast.unparse(x) for x in methods)
s+="""
openids=t.encode(xml.rsplit('</arg_value>',1)[0],add_special_tokens=False)
checks=0
for bad in [154844,154820,154827,154829]:
 for keep in range(7):
  req=newreq('A');req.is_finished=lambda:False;req.is_prefill_chunk=False
  commit(req,openids[:-keep] if keep else openids)
  draft=(openids[-keep:] if keep else [])+[bad]+[0]*(6-keep)
  sched=SimpleNamespace(requests={'A':req},structured_output_manager=manager)
  output=SimpleNamespace(scheduled_spec_decode_tokens={'A':[-1]*7},has_structured_output_requests=True,num_scheduled_tokens={'A':8})
  update_draft_token_ids_in_output(sched,SimpleNamespace(req_ids=['A'],draft_token_ids=[draft]),output)
  assert output.scheduled_spec_decode_tokens['A'][keep:]==[-1]*(7-keep)
  grammar=get_grammar_bitmask(sched,output)
  assert grammar.num_acceptable_drafts==[keep]
  assert not allows(grammar.grammar_bitmask[keep],bad)
  # Rejected proposal must not advance persistent grammar; valid replacement
  # delimiter commits, then EOS cannot stop before tool closer.
  closer=t.encode('</arg_value>',add_special_tokens=False)
  commit(req,output.scheduled_spec_decode_tokens['A'][:keep]+closer)
  assert manager.validate_tokens(req,[154820])==[]
  checks+=1
print('ACTUAL_SCHEDULER_HANDOFF_CASES',checks,'CUDA',torch.cuda.is_initialized(),flush=True)
"""
(d/'scheduler_probe.py').write_text(s)
