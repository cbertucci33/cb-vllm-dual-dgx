"""Dependency-free checks of selected production source; not stack acceptance."""
import ast
import json
import pathlib

import pytest

ROOT = pathlib.Path('/opt/project/builds/glm-mimo-main-fixes-20261017')


def production_symbols(file, names):
    tree = ast.parse((ROOT / file).read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    scope = {'json': json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])), file, 'exec'), scope)
    return scope


@pytest.mark.parametrize('parameters_first', [False, True])
@pytest.mark.parametrize('replay', ['whole', 'characters', 'every_split'])
def test_required_json_original_bytes_chunk_invariance(parameters_first, replay):
    scanner = production_symbols('vllm/parser/json_tools.py', ['_RequiredJSONStream'])['_RequiredJSONStream']
    arguments = [json.dumps({'text': '漢字 <think>quoted</think> \\"\n', 'nested': [True, None, {'x': 1}]}, ensure_ascii=False), '{ "x" : [1, 2] }']
    envelopes = []
    for name, raw in zip(['echo', 'second'], arguments):
        pair = ['"name":' + json.dumps(name), '"parameters":' + raw]
        if parameters_first:
            pair.reverse()
        envelopes.append('{' + ','.join(pair) + '}')
    text = '[' + ','.join(envelopes) + ']'
    chunkings = [[text]] if replay == 'whole' else [list(text)] if replay == 'characters' else [[text[:i], text[i:]] for i in range(len(text)+1)]
    for chunks in chunkings:
        instance = scanner()
        seen = {}
        first_counts = {}
        for chunk in chunks:
            for index, name, raw, first in instance.feed(chunk):
                seen[index] = (name, seen.get(index, ('', ''))[1] + raw)
                first_counts[index] = first_counts.get(index, 0) + int(first)
        assert seen == {0: ('echo', arguments[0]), 1: ('second', arguments[1])}
        assert first_counts == {0: 1, 1: 1}
        assert instance.feed('') == []


@pytest.mark.parametrize('raw', ['{"text":"unfinished', '{"text":"\\u', '{"x":[1,'])
def test_required_json_does_not_invent_truncated_suffix(raw):
    scanner = production_symbols('vllm/parser/json_tools.py', ['_RequiredJSONStream'])['_RequiredJSONStream']()
    updates = scanner.feed('[{"name":"echo","parameters":' + raw)
    assert ''.join(x[2] for x in updates) == raw
    assert scanner.feed('') == []


def test_non_harmony_does_not_synthesize_phase_or_buffer_done():
    for file in ('context.py', 'utils.py', 'serving.py', 'streaming_events.py'):
        text = (ROOT / 'vllm/entrypoints/openai/responses' / file).read_text()
        assert 'response_message_phase' not in text
        assert 'SimpleMessagePhaseResolver' not in text
        tree = ast.parse(text)
        assert not any(isinstance(n, ast.keyword) and n.arg == 'phase' for n in ast.walk(tree))


def test_reasoning_wrappers_use_standard_content_part_events():
    tree = ast.parse((ROOT / 'vllm/entrypoints/openai/responses/streaming_events.py').read_text())
    for name in ('emit_reasoning_delta_events', 'emit_reasoning_done_events', 'emit_simple_reasoning_open', 'emit_simple_reasoning_done'):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        constants = {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert constants & {'response.content_part.added', 'response.content_part.done'}
        assert not constants & {'response.reasoning_part.added', 'response.reasoning_part.done'}


@pytest.mark.parametrize('template, expected', [
    ('[gMASK]<sop> Reasoning Effort: <tool_call><arg_key><arg_value>', True),
    ('[gMASK]<sop> Reasoning Effort: <tool_call><arg_key><arg_value> enable_thinking', False),
    ('Reasoning Effort: <tool_call><arg_key><arg_value>', False),
    (None, False),
])
def test_glm53_template_contract_has_one_production_definition(template, expected):
    from types import SimpleNamespace
    file = 'vllm/parser/glm47_moe.py'
    tree = ast.parse((ROOT / file).read_text())
    assert sum(isinstance(n, ast.FunctionDef) and n.name == '_glm53_always_thinks' for n in tree.body) == 1
    func = production_symbols(file, ['_glm53_always_thinks'])['_glm53_always_thinks']
    assert func(SimpleNamespace(chat_template=template)) is expected
