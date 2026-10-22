from pathlib import Path
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017')
p=r/'vllm/model_executor/layers/quantization/exl3.py';s=p.read_text();a=s.index('def _exl3_gemm(');b=s.index('@_exl3_gemm.register_fake',a);f=s[a:b]
assert 'dtype=torch.float16' in f and '    return output\n' in f
f=f.replace('dtype=torch.float16','dtype=torch.float32',1).replace('    return output\n','    # C also stores inter-CTA split-K partial sums in the native kernel.\n    return output.to(torch.float16)\n',1)
p.write_text(s[:a]+f+s[b:])
p=r/'tests/quantization/test_exl3.py';s=p.read_text();assert 'def test_exl3_gemm_retains_fp32_native_accumulation_and_fp16_public_output' not in s
s+='''\n\ndef test_exl3_gemm_retains_fp32_native_accumulation_and_fp16_public_output():
    """Static wrapper contract; this does not qualify native numeric behavior."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(exl3_module.__file__).read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    function = functions["_exl3_gemm"]
    output_allocation = next(
        node.value
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "output" for target in node.targets)
    )
    assert ast.unparse(output_allocation.func) == "torch.empty"
    dtype = next(keyword.value for keyword in output_allocation.keywords if keyword.arg == "dtype")
    assert ast.unparse(dtype) == "torch.float32"
    native_call = next(
        node.value
        for node in function.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "ext.exl3_gemm"
    )
    assert len(native_call.args) == 10
    assert ast.unparse(native_call.args[2]) == "output"
    result = next(node.value for node in function.body if isinstance(node, ast.Return))
    assert ast.unparse(result) == "output.to(torch.float16)"
    fake_result = next(
        node.value for node in functions["_exl3_gemm_fake"].body if isinstance(node, ast.Return)
    )
    fake_dtype = next(keyword.value for keyword in fake_result.keywords if keyword.arg == "dtype")
    assert ast.unparse(fake_dtype) == "torch.float16"
'''
p.write_text(s)
