
import json, os, shutil, re, random, io, time
import torch

# ---- 端口配置（单一来源，grpo_vllm_one.py 的客户端会 import 这两个值并扫描同样的范围）----
# 端口被占用时，ref_server 会自动往后顺延；最多顺延 max_port_retries 次，每次顺延打一条 warning，
# 全部都被占用才会报错。客户端 (grpo_vllm_one.py) 会跟着扫描这个范围，所以改这里一处即可。
base_port = 59875
max_port_retries = 3

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()
def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)
def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()
def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

if __name__ == '__main__':   
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn as nn

    from bottle import request
    import bottle, threading, queue
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    model_path = "./models/Qwen/Qwen2.5-3B"

    ref_model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
    ref_model.eval()
    ref_model.requires_grad_(False)

    def get_per_token_logps(input_ids):
        logits = ref_model(input_ids).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    # 默认 maxsize=0，也就是无界；Lifo 的意思是，get 时永远取最后放进去的那条（后进先出），从而达到 on-policy RL 想要的效果：最新策略分布生成的数据先用
    raw_queue = queue.LifoQueue()
    result_queue = queue.LifoQueue()

    app = bottle.Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        """
        gen worker 使用，接受其POST 来的 batch，拆解后塞进 raw_queue, 等主循环算 log-probs
        """
        dd = request.body.read()
        dd = bytes_list_to_list(dd)
        if len(dd) not in (3,4): return b'tensor'
        data = {'base': json.loads(dd[0])} 
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        if len(dd) == 4: data['gen_logps'] = bytes_to_tensor(dd[3])
        raw_queue.put(data)
        print('receive', data['inputs'].shape, data['rewards'], 
              data['gen_logps'].shape if 'gen_logps' in data else '')
        return b'tensor'

    @app.route('/get', method='GET')
    def do_get():
        """
        训练进程使用，轮询拿数据；空则阻塞
        """
        if result_queue.empty(): return b'empty'
        return result_queue.get()
    
    def run_server():
        """
        用独立线程跑 bottle server，避免阻塞主循环。
        端口被占用时自动往后顺延（最多 max_port_retries 次），每次顺延打一条 warning；
        全部端口都被占用才抛 RuntimeError。客户端 (grpo_vllm_one.py) 会扫描同样的范围跟上。
        这个进程真正做的事在 while True: 循环里。

        注意：这里没有用 bottle.run(server='tornado')，而是手动搭 tornado —— 因为
        bottle 的 TornadoServer 适配器内部直接 server.listen(port)，端口被占用就抛异常退出，
        没法捕获。手动搭一遍就能在 listen() 外面套 try，从而实现顺延。
        """
        import tornado.wsgi, tornado.httpserver, tornado.ioloop
        container = tornado.wsgi.WSGIContainer(app)
        bound_port = None
        for offset in range(max_port_retries + 1):
            port = base_port + offset
            try:
                server = tornado.httpserver.HTTPServer(container)
                server.listen(port=port, address='0.0.0.0')
                bound_port = port
                break
            except OSError as e:
                reason = e.strerror or str(e)
                if offset < max_port_retries:
                    print(f"[warning] ref_server 端口 {port} 被占用（{reason}），顺延到 {port + 1} ...")
                else:
                    print(f"[warning] ref_server 端口 {port} 被占用（{reason}），已用尽 {max_port_retries + 1} 次尝试")
        if bound_port is None:
            raise RuntimeError(
                f"ref_server 在端口 {base_port}~{base_port + max_port_retries} 全部被占用，无法启动；"
                f"请清理占用进程或调大 max_port_retries")
        print(f"[ref_server] listening on http://0.0.0.0:{bound_port}")
        tornado.ioloop.IOLoop.instance().start()
    threading.Thread(target=run_server, daemon=False).start()

    while True:
        """
        计算 ref model 的 log-probs 并返回给训练进程
        训练进程会把数据塞进 raw_queue, 这个循环从 raw_queue 里拿数据，计算 log-probs 后塞进 result_queue
        """
        d = raw_queue.get()
        prompt_length = d['base']['plen']
        with torch.inference_mode():
            # 获取 ref model 对每个 token 的 log-probs
            per_token_logps = get_per_token_logps(d['inputs'].to(ref_model.device))
        per_token_logps = per_token_logps[:,prompt_length-1:]
        data = [json.dumps(d['base']).encode(), tensor_to_bytes(d['inputs']), 
                tensor_to_bytes(d['rewards']), tensor_to_bytes(per_token_logps)]
        if 'gen_logps' in d: data.append(tensor_to_bytes(d['gen_logps']))
        xdata = make_bytes_list(data)
        result_queue.put(xdata)
