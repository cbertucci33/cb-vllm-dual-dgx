import ast,pathlib,re,subprocess
r=pathlib.Path('/opt/project/builds/glm-mimo-main-fixes-20261017'); out=pathlib.Path('/opt/project/evidence/vllm-v032-migration-20261018/parser')
prefixes=('vllm/parser/','vllm/tool_parsers/','vllm/reasoning/','vllm/tokenizers/','vllm/entrypoints/','tests/parser/','tests/reasoning/','tests/tool_parsers/','tests/tool_use/','tests/entrypoints/')
owned=[f for f in subprocess.check_output(['git','-C',str(r),'diff','--name-only','--diff-filter=U'],text=True).splitlines() if f.startswith(prefixes) or f=='tests/v1/structured_output/test_structured_output_manager.py']
log=[]
for f in owned:
 p=r/f;s=p.read_text();n=0
 def resolve(m):
  nonlocal_dummy=None
  a,b=m.group(1),m.group(2);i=len([x for x in log if x[0]==f]);choice='upstream'
  if not a.strip():c=b;choice='local addition'
  elif not b.strip():c=a
  elif f=='vllm/entrypoints/mcp/tool.py':c=b;choice='local multipart join'
  elif f=='vllm/entrypoints/openai/responses/context.py':c='from vllm.entrypoints.mcp.tool import Tool, mcp_result_to_tool_output\n';choice='local rich result helper + upstream renderer import below'
  elif f=='vllm/entrypoints/openai/responses/serving.py':
   if 'reuse_streamed_item_ids' in a:c=a
   else:c=b;choice='local parse_deltas plus upstream filtering added separately'
  elif f=='vllm/entrypoints/openai/responses/streaming_events.py':
   if 'tool_call_is_custom' in a:c=a+'    has_emitted_tool_call_delta: bool = False\n';choice='union state; upstream logprob field already declared'
   else:c=a;choice='upstream custom tools and output-logprob contract'
  elif f=='vllm/entrypoints/openai/responses/utils.py':
   if 'from partial_json_parser' in a:c=a+b;choice='union imports'
   elif 'def custom_tool_names' in a:c=a+b;choice='union upstream custom tools and local rich outputs; phase inference removed separately'
   else:c=b.replace('item.get','output.get');choice='local rich conversion with upstream output variable'
  elif f=='vllm/parser/engine/parser_engine.py':c=b+'from vllm.renderers.chat_utils import get_tool_call_id_type, make_tool_call_id\n';choice='local extender/token accounting + upstream renderer import'
  elif f=='vllm/parser/mimo.py':c=b;choice='local unified MiMo parser; renderer import corrected separately'
  elif f=='vllm/tokenizers/detokenizer_utils.py':c='    new_text = new_text[prefix_length:]\n    return new_tokens, new_text, read_offset, output_token_count\n';choice='local incomplete-UTF8 boundary plus upstream bounded-window absolute count'
  elif f=='tests/entrypoints/openai/responses/test_streaming_events.py':
   c=a+('from openai.types.responses import response_text_delta_event\n' if 'import pytest' in a else b);choice='union tests/imports'
  elif f=='tests/tool_parsers/test_structural_tag_registry.py' and 'BEGIN_TOOL_ARGUMENTS' in a:c=a+b;choice='union upstream Plamo tests and local MiMo registrations'
  else:c=a
  log.append((f,i+1,choice));return c
 s=re.sub(r'<<<<<<< ours\n(.*?)=======\n(.*?)>>>>>>> theirs\n',resolve,s,flags=re.S)
 assert '<<<<<<<' not in s,f
 p.write_text(s)
# Fix module moves in carried parser/tests; old module was removed upstream.
for p in [r/'vllm/parser/mimo.py',r/'vllm/parser/glm47_moe.py',r/'tests/entrypoints/openai/responses/test_responses_utils.py',r/'tests/entrypoints/openai/responses/test_parsable_context_unit.py']:
 s=p.read_text().replace('from vllm.entrypoints.chat_utils import','from vllm.renderers.chat_utils import');p.write_text(s)
# Remove unsupported phase inference, leaving Harmony channels untouched.
for f in ['vllm/entrypoints/openai/responses/utils.py','vllm/entrypoints/openai/responses/serving.py','vllm/entrypoints/openai/responses/context.py','vllm/entrypoints/openai/responses/streaming_events.py']:
 p=r/f;s=p.read_text();s=s.replace('    response_message_phase,\n','').replace('    SimpleMessagePhaseResolver,\n','')
 s=re.sub(r'^\s*phase=response_message_phase\([^\n]*\),\n','\n',s,flags=re.M)
 if f.endswith('utils.py'):s=re.sub(r'\ndef response_message_phase\(.*?(?=\ndef )','\n',s,flags=re.S)
 if f.endswith('streaming_events.py'):s=s[:s.index('\nclass SimpleMessagePhaseResolver:')].rstrip()+'\n'
 if f.endswith('serving.py'):
  s=re.sub(r'^\s*phases = SimpleMessagePhaseResolver\(\)\n','\n',s,flags=re.M)
  s=re.sub(r'        (    )?for event in record_events\(phases.finish\(finish_reason\)\):\n            (    )?yield event\n','',s)
  tree=ast.parse(s);lines=s.splitlines(keepends=True);offsets=[0]
  for l in lines:offsets.append(offsets[-1]+len(l))
  edits=[]
  for node in ast.walk(tree):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='phases' and node.func.attr=='process':
    edits.append((offsets[node.lineno-1]+node.col_offset,offsets[node.end_lineno-1]+node.end_col_offset,ast.get_source_segment(s,node.args[0])))
  for a,b,v in sorted(edits,reverse=True):s=s[:a]+v+s[b:]
  needle='                part for message in delta_messages for part in split_delta(message)'
  assert needle in s
  s=s.replace(needle,'                part\n                for message in delta_messages\n                for part in split_delta(\n                    maybe_filter_parallel_tool_calls(message, request)\n                )')
 p.write_text(s)
# Keep useful lifecycle fixture/tests; revise invalid phase assertions to unset.
p=r/'tests/entrypoints/openai/responses/test_message_phases.py';s=p.read_text();tree=ast.parse(s);lines=s.splitlines(keepends=True);offs=[0]
for l in lines:offs.append(offs[-1]+len(l))
edits=[]
for node in ast.walk(tree):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='assertEqual' and node.args:
  first=ast.get_source_segment(s,node.args[0])
  if '.phase' in first:
   arg=node.args[1];edits.append((offs[arg.lineno-1]+arg.col_offset,offs[arg.end_lineno-1]+arg.end_col_offset,'None' if not first.startswith('[') else '[None for _ in '+first+']'))
 if isinstance(node,ast.Assert):
  text=ast.get_source_segment(s,node.test)
  if text=='[m.phase for m in visible] == ["commentary", "final_answer"]':edits.append((offs[node.test.lineno-1]+node.test.col_offset,offs[node.test.end_lineno-1]+node.test.end_col_offset,'[m.phase for m in visible] == [None, None]'))
for a,b,v in sorted(edits,reverse=True):s=s[:a]+v+s[b:]
s=s.replace('CPU-only contracts for non-Harmony Responses message delivery phases.','CPU-only contracts for non-Harmony Responses item lifecycle.\n\nNon-Harmony text has no model-provided channel: tool presence and finish reason\nmust not synthesize commentary/final-answer phases.')
p.write_text(s)
(out/'conflict-decisions.tsv').write_text('path\tconflict\tchoice\n'+''.join(f'{f}\t{n}\t{c}\n' for f,n,c in log))
(out/'owned-conflicts.txt').write_text('\n'.join(owned)+'\n')
print('resolved',len(owned),'paths',len(log),'conflict blocks')
