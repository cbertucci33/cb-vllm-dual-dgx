import pathlib,json,struct,collections
root=pathlib.Path('/opt/project/evidence/vllm-v032-migration-20261019/mimo-qualification-preparation');p=pathlib.Path('/opt/project/models/mimo-heretic-v031-mm-b12f5573-targetkv-cdb36376ae25-20261013');h={}
for f in p.glob('*.safetensors'):
 with f.open('rb') as s:d=json.loads(s.read(struct.unpack('<Q',s.read(8))[0]))
 h.update({k:v for k,v in d.items() if k!='__metadata__'})
rows=[]
for k,t in sorted(h.items()):
 if not k.endswith('.trellis') or '.rank' in k:continue
 name=k[:-8]; fields={f:{'shape':h[name+'.'+f]['shape'],'bytes':h[name+'.'+f]['data_offsets'][1]-h[name+'.'+f]['data_offsets'][0]} for f in ['trellis','suh','svh','mcg']};total=sum(x['bytes'] for x in fields.values())
 if '.mtp.' in name:rule='excluded_mtp'; consumer='mimo_v2.py:762 skips mtp';resident=0
 elif name.endswith('.o_proj') or name.endswith('.down_proj'):rule='row_contiguous_view_retains_full_backing';consumer='RowParallelLinear';resident=total
 elif name.endswith('.v_proj'):rule='padded_V_replicated';consumer='QKVParallelLinear[v], exl3.py:1017/1036';resident=total
 else:
  assert name=='lm_head' or name.endswith(('.q_proj','.k_proj','.gate_proj','.up_proj')),name
  rule='column_trellis_compact_copy_sign_views_retain_full';consumer='ParallelLMHead' if name=='lm_head' else 'QKVParallelLinear' if name.endswith(('.q_proj','.k_proj')) else 'MergedColumnParallelLinear';resident=total-fields['trellis']['bytes']//2
 rows.append({'matrix':name,'fields':fields,'consumer':consumer,'rule':rule,'pre_postprocess_destination_bytes':0 if rule=='excluded_mtp' else total,'retained_storage_bytes_per_rank':resident,'column_copy_extra_bytes':fields['trellis']['bytes']//2 if rule.startswith('column') else 0,'source_anchors':['mimo_v2.py:89-102,264-281,757-765,964-971','exl3.py:706-717,783,952-965,992-1054']})
assert len(rows)==220
out={'denominator':220,'classified':len(rows),'rules':dict(collections.Counter(r['rule'] for r in rows)),'active_pre_postprocess_dense_bytes':sum(r['pre_postprocess_destination_bytes'] for r in rows),'active_retained_dense_bytes':sum(r['retained_storage_bytes_per_rank'] for r in rows),'max_individual_column_copy_bytes':max(r['column_copy_extra_bytes'] for r in rows),'matrices':rows}
(root/'dense-matrix-inventory.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='matrices'},indent=2))
