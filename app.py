from flask import Flask, request, render_template, jsonify
import requests
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
import numpy as np
from llama_cpp import Llama
import os

app = Flask(__name__)

# ===== 設定 =====
LLAMA_MODEL_PATH = "models/Phi-3.5-mini-instruct-Q4_0.gguf" # Please change to Your GGUF File.
llm = Llama(model_path=LLAMA_MODEL_PATH, n_threads=4)
rerank_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ===== ユーティリティ関数 =====
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def rerank_results(query, results):
    query_embedding = rerank_model.encode([query])[0]
    doc_texts = [r['body'] + " " + r['title'] for r in results]
    doc_embeddings = rerank_model.encode(doc_texts)
    scores = [cosine_similarity(query_embedding, emb) for emb in doc_embeddings]
    sorted_results = [x for _, x in sorted(zip(scores, results), key=lambda x: x[0], reverse=True)]
    return sorted_results, scores

def llama_completion(prompt, max_tokens=8192, temperature=0.2):
    output = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["</s>","###"],
    )
    text = output["choices"][0]["text"].strip()
    return text

def summarize_content(contents):
    prompt = f"Below you will be tasked with creating a summary of search results from the web. Please consider the key points and summarize them.<br><br>---<br>{contents}<br>---<br>Summary: "
    summary = llama_completion(prompt)
    return summary

def answer_question(question, context):
    prompt = f"Below is a summary of the search results.<br><br>---<br>{context}<br>---<br><br>Please use this as a reference to answer the following questions.\nQuestion: {question}<br>Answer: "
    answer = llama_completion(prompt).split("Question:")[0]
    return answer

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')
    ddgs = DDGS()
    results = list(ddgs.text(query, max_results=10))
    reranked_results, scores = rerank_results(query, results)

    # トップ3件で要約
    top_docs = reranked_results[:3]
    combined_text = "\n".join([f"Title: {doc['title']}\nURL: {doc['href']}\nContent: {doc['body']}" for doc in top_docs])
    summary = summarize_content(combined_text)

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            "query": query,
            "results": reranked_results,
            "summary": summary
        })
    else:
        # fallback
        return render_template('index.html', query=query, results=reranked_results, summary=summary)

@app.route('/answer', methods=['POST'])
def get_answer():
    query = request.form.get('query', '')
    research_question = request.form.get('research_question', '')
    # 再検索してsummaryとtop docsのテキストを得る
    ddgs = DDGS()
    results = list(ddgs.text(query, max_results=10))
    reranked_results, scores = rerank_results(query, results)
    top_docs = reranked_results[:3]
    combined_text = "\n".join([f"Title: {doc['title']}\nURL: {doc['href']}\nContent: {doc['body']}" for doc in top_docs])

    answer = answer_question(research_question, combined_text)

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            "answer": answer
        })
    else:
        # fallback
        summary = summarize_content(combined_text)
        return render_template('index.html', 
                               query=query, 
                               results=reranked_results, 
                               summary=summary, 
                               research_question=research_question,
                               answer=answer)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
