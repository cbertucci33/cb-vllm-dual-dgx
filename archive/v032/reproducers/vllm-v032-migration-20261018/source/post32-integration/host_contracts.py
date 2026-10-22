"""Execute production host bookkeeping only; recording substitutes are NOT GPU tests."""
import ast
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path('/opt/project/builds/glm-mimo-main-fixes-20261017')
EVIDENCE = Path(__file__).parent
results = []

def load(path, before=False):
    return ast.parse(((EVIDENCE / 'before' / path) if before else ROOT / path).read_text())

def execute(nodes, env):
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), '<production AST>', 'exec'), env)
    return env

class Buffer:
    def __init__(self, owner=None, stop=None):
        self.owner = owner or self
        self.stop = stop
        if owner is None:
            self.zeros = []
        self.shape = (8, 16)
    def __getitem__(self, key):
        return Buffer(self.owner, key.stop)
    def zero_(self):
        self.owner.zeros.append(self.stop)

class Kernel:
    def __init__(self):
        self.calls = []
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return launch

def block_methods(before=False):
    tree = load('vllm/v1/worker/gpu/block_table.py', before)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BlockTables')
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('gather_block_tables', 'get_dummy_block_tables')]
    kernel = Kernel()
    env = execute(nodes, {'_gather_block_tables_kernel': kernel})
    return env, kernel

def subject():
    return NS(num_kv_cache_groups=2, input_block_tables=[Buffer(), Buffer()], input_block_table_ptrs='shared-ptrs', block_table_ptrs='source-ptrs', block_table_strides='strides', num_blocks=NS(gpu=NS(stride=lambda axis: 8)), _num_dirty_rows=8)

for previous_dummy, next_dummy in itertools.product((False, True), repeat=2):
    env, kernel = block_methods()
    obj = subject()
    for count in (3, 5, 0):
        dirty = obj._num_dirty_rows
        if next_dummy:
            out = env['get_dummy_block_tables'](obj, count)
            assert obj.input_block_tables[0].zeros[-1] == max(count, dirty)
        else:
            out = env['gather_block_tables'](obj, NS(shape=(count,)), count)
            assert kernel.calls[-1][0] == (2, max(count, dirty))
            assert kernel.calls[-1][1][-1] == count
        assert obj._num_dirty_rows == count
        assert all(view.owner is owner and view.stop == count for view, owner in zip(out, obj.input_block_tables))
        # Previous dummy profiling writes may occupy the whole returned range.
        if previous_dummy:
            env['get_dummy_block_tables'](obj, count)
    results.append(f'shared shrink lifecycle previous_dummy={previous_dummy},next_dummy={next_dummy}')
env, kernel = block_methods()
obj = subject()
external = (Buffer(), Buffer())
out = env['gather_block_tables'](obj, NS(shape=(2,)), 4, external, 'external-ptrs')
assert obj._num_dirty_rows == 8 and kernel.calls[-1][0] == (2, 4)
assert kernel.calls[-1][1][2] == 'external-ptrs'
assert all(v.owner is b for v, b in zip(out, external))
results.append('external output preserves shared dirty ownership')
obj.num_kv_cache_groups = 0
assert env['gather_block_tables'](obj, NS(shape=(2,)), 4) == ()
results.append('zero cache groups early return')
env_old, kernel_old = block_methods(True)
old_obj = subject()
env_old['gather_block_tables'](old_obj, NS(shape=(3,)), 3)
assert kernel_old.calls[-1][0] == (2, 3), 'before fixture no longer discriminates'
assert kernel_old.calls[-1][0] != (2, 8)
results.append('before version fails larger shared-row launch regression')

class Rows:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shape = (len(rows), 1)
        self.dtype = 'int32'
    def __getitem__(self, key):
        return Rows(self.rows[key]) if isinstance(key, slice) else self.rows[key]
    def __setitem__(self, key, value):
        self.rows[key] = value.rows if isinstance(value, Rows) else value
    def numpy(self):
        return self

def reorder(before=False):
    tree = load('vllm/v1/structured_output/utils.py', before)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'apply_grammar_bitmask')
    body = []
    for node in fn.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'grammar_bitmask' for t in node.targets) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'to':
            break
        body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    grammar = NS(grammar_bitmask=Rows([10,11,12,20,21,22,30,31,32]), structured_output_request_ids=['missing','b','a'])
    env = {'grammar_output': grammar, 'scheduler_output': NS(scheduled_spec_decode_tokens={'b':[1]}), 'input_batch': NS(req_ids=['a','plain','b']), 'logits': NS(shape=(4,)), 'PIN_MEMORY':False, 'torch': NS(from_numpy=lambda x:x, full=lambda shape,value,**kwargs:Rows([value]*shape[0]))}
    exec(compile(ast.fix_missing_locations(module), '<production MRV1 prefix>', 'exec'),env)
    return env['sorted_bitmask'].rows, env['out_indices']
assert reorder() == ([30,-1,20,21],[2,3,0])
assert reorder(True)[0] != [30,-1,20,21]
results.extend(['MRV1 missing/reordered request fixed-block cursor', 'before MRV1 selects wrong fixed block'])

tree = load('vllm/v1/worker/gpu/structured_outputs.py')
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StructuredOutputsWorker')
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'apply_grammar_bitmask')
assignment = next(n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'mapping' for t in n.targets))
for stride in (1,3,8):
    for request_ids in itertools.permutations(('a','b','c')):
        mapping = eval(compile(ast.Expression(assignment.value), '<production mapping>', 'eval'), {'req_id_to_idx':{v:i for i,v in enumerate(request_ids)}, 'self':NS(mask_stride=stride), 'grammar_req_ids':['c','a']})
        assert mapping == [request_ids.index(r)*stride+p for r in ('c','a') for p in range(stride)]
results.append('MRV2 mapping 18 request-permutation/stride cases')
launch = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func,ast.Subscript) and isinstance(n.func.value,ast.Name) and n.func.value.id == '_apply_grammar_bitmask_kernel')
kernel_fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == '_apply_grammar_bitmask_kernel')
assert len(launch.args) == 7 and {k.arg for k in launch.keywords} == {'MASK_STRIDE','BLOCK_SIZE'}
assert len(kernel_fn.args.args) == 9
assert ast.unparse(launch.args[3]) == 'input_batch.cu_num_logits'
store = next(n for n in ast.walk(kernel_fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr == 'store')
assert any('position_is_active' in ast.unparse(k.value) for k in store.keywords if k.arg == 'mask')
assert any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'position_is_active' for t in n.targets) and ast.unparse(n.value) == 'position_idx < num_req_logits' for n in kernel_fn.body)
results.append('native launch binding and device active-position store guard source contract')
for path in json.loads((EVIDENCE/'owned-paths.json').read_text()):
    ast.parse((ROOT/path).read_text())
results.append('12 owned Python paths AST parse')
report = {'passed_checks':len(results), 'checks':results, 'mapping_cases':18, 'limitations':['Recording substitutes validate host launch/ownership bookkeeping only, not tensor writes or GPU correctness.', 'Native Triton/xgrammar execution, real Torch/NumPy tests, graph replay and model performance unverified.']}
(EVIDENCE/'host-contract-results.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
