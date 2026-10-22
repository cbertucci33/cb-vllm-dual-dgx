from pathlib import Path
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017')
p=r/'tests/models/glm5next/test_dflash_aux_hidden.py';s=p.read_text().replace('    model.is_sequence_parallel = False','    model.start_layer = 0\n    model.end_layer = 3\n    model.is_sequence_parallel = False').replace('    model.is_sequence_parallel = True','    model.start_layer = 0\n    model.end_layer = 3\n    model.is_sequence_parallel = True')
s=s.replace('        captured = model._completed_layer_output(\n            Layer(), hidden_states, residual, post, comb\n        )','        model.config = SimpleNamespace(hc_mult=2)\n        model._aux_post_op = Layer.hc_post\n        captured = model._aux_hidden_state(hidden_states, residual, post, comb)')
s=s.replace('test_glm5_sequence_parallel_gathers_final_then_aux_in_capture_order','test_glm5_sequence_parallel_preserves_completed_aux_and_final_outputs')
s=s.replace('torch.testing.assert_close(gathered[0], inputs + 3)\n    torch.testing.assert_close(gathered[1], inputs + 1)','torch.testing.assert_close(gathered[0], inputs + 1)\n    torch.testing.assert_close(gathered[1], inputs + 3)');p.write_text(s)
p=r/'tests/v1/spec_decode/test_dflash_graph_context.py';s=p.read_text().replace('spec._prepare_graph_forward(2)','spec._prepare_graph_forward(2, None)');p.write_text(s)
p=r/'tests/v1/spec_decode/test_dflash_model_policy.py';s=p.read_text().replace('graph_context = lambda num_reqs: None','graph_context = lambda num_reqs, attn_metadata: None').replace('_kwargs["precompute_context_kv"]','_kwargs["prepare_forward"]');p.write_text(s)
p=r/'build/README.md';s=p.read_text().replace('0.31','0.32').replace('4.7.1','4.8.0').replace('0.1.16','0.1.17').replace('dc04f50c9aa3eabcdaa5feb0934edb3d85e9529a','946200de1ae94fc93fdd0926f0a13afd1fa7f0f1')
s=s.replace('A source-built `0.7.0` distribution is accepted in place of official\n`0.7.0.post1` only when','The source-built `0.7.0.post1` distribution is verified only when')
s='''> Migration status: source-only .32 integration. No .32 native build or runtime acceptance has been performed. Existing base-image digest examples below are historical operands, not a newly qualified .32 build.\n\n'''+s
p.write_text(s)
