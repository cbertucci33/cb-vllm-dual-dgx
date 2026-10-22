import json,time,urllib.request,pathlib
p=pathlib.Path(__file__).parent
base="http://127.0.0.1:8000"
def call(tag,body):
 body.update(model="model-main",temperature=1.0,top_p=.95,reasoning={"effort":"high"},stream=False)
 (p/(tag+"-request.json")).write_text(json.dumps(body));t=time.monotonic()
 with urllib.request.urlopen(urllib.request.Request(base+"/v1/responses",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=180) as r: out=json.load(r)
 (p/(tag+"-response.json")).write_text(json.dumps(out,indent=2));print(tag,round(time.monotonic()-t,3),out.get("status"),out.get("usage"),flush=True)
 assert out.get("status")=="completed",out
 return out
# Known text prefix long enough for required1024-token draft replay, bounded ordinary request.
prefix="Reference facts: the observation code is ORCHID and the answer to seventeen plus twenty-five is forty-two. "*110
body={"input":[{"role":"user","content":prefix+"Reply only ORCHID."}]}
for tag in ["prefix-first","prefix-reuse"]:
 out=call(tag,body.copy()); txt="".join(c.get("text","") for i in out["output"] if i["type"]=="message" for c in i["content"]);assert txt.strip()=="ORCHID",txt
assert out["usage"]["input_tokens_details"]["cached_tokens"]>0,out["usage"]
tools=[{"type":"function","name":"lookup_test_value","description":"Retrieve the test value for the supplied key.","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"],"additionalProperties":False}}]
inp=[{"role":"user","content":"Call lookup_test_value with key ORCHID. After its result, reply only with the returned value."}]
out=call("tool",{"input":inp,"tools":tools}); fc=[i for i in out["output"] if i["type"]=="function_call"];assert len(fc)==1 and fc[0]["name"]=="lookup_test_value";assert json.loads(fc[0]["arguments"])=={"key":"ORCHID"}
continuation=inp+out["output"]+[{"type":"function_call_output","call_id":fc[0]["call_id"],"output":"42"}]
out=call("tool-continuation",{"input":continuation,"tools":tools});txt="".join(c.get("text","") for i in out["output"] if i["type"]=="message" for c in i["content"]);assert txt.strip()=="42",txt
