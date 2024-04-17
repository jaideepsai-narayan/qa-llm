from final import qa_chain
import gc
import torch
from flask import Flask, render_template,render_template_string,request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    code_snippet_output= qa_chain.invoke(userText)['result'].split("Helpful Answer:")[-1]
    def flush():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    flush()
    if "code" in userText:
        return render_template_string("<div style='overflow-x: auto; white-space: pre-wrap;'><pre>{{ code_snippet_output }}</pre></div>", code_snippet_output=code_snippet_output)
    else:
        return code_snippet_output

app.run(debug = True)