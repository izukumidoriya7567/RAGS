"""
Two modes are provided:

  MODE A: use_builtin_chat()
      Uses PageIndex's own hosted Chat API (`chat_completions`).
      Simplest option — PageIndex handles tree reasoning + retrieval +
      answer generation for you. No external LLM key needed.

  MODE B: custom_pipeline()
      Does the "index -> reason over tree -> pull only relevant node text
      -> generate answer" pipeline yourself, using your own LLM (here:
      the Anthropic API). Use this when you want control over the model,
      the prompt, or want to see exactly which sections were used.
"""

import os
import time
import json
from pageindex import PageIndexClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PAGEINDEX_API_KEY = os.getenv("PAGE_INDEX_API_KEY")
DOC_PATH = "./r1200appendix_d.pdf"

pageIndexClient = PageIndexClient(api_key=PAGEINDEX_API_KEY)

def submitAndWait(docPath:str|None=None, pollSeconds:int=5)-> str:
    if docPath is None:
        raise ValueError("docPath must be provided.")
    result = pageIndexClient.submit_document(docPath)
    docId = result["doc_id"]

    while True:
        status = pageIndexClient.get_document(docId)["status"]
        if status=="completed":
            print("PageIndex, DocId-:", docId)
            print("Document Processing Completed")
            break
        elif status=="failed":
            break
        else:
            print("Still processing... checking again in", pollSeconds, "s")
        time.sleep(pollSeconds)

    return docId


# ---------------------------------------------------------------------------
# MODE A: PageIndex's own hosted Chat API does indexing + retrieval + answer
# ---------------------------------------------------------------------------

def useBuiltInChat(docId:str, question:str)-> str:
    if docId is None:
        raise ValueError("docId must be provided.")
    
    response = pageIndexClient.chat_completions(
        doc_id=docId,
        messages=[{"role": "user", "content": question}],
    )
    return response["choices"][0]["message"]["content"]


# MODE B: custom pipeline — you control the LLM and see the retrieved nodes
# ---------------------------------------------------------------------------

def flattenTree(nodes, out=None):
    if out is None:
        out=[]
    
    for node in nodes:
        out.append(node)
        children = node.get("nodes") or node.get("children")
        if children:
            flattenTree(children, out)
    
    return out

def buildSummaryOutline(flatNodes):
    lines = []

    for node in flatNodes:
        print("Node Details:",node)
        print("----------------END----------------")

        nodeId=node.get("node_id") or node.get("id")
        title=node.get("title","")
        summary=node.get("summary","")
        print("Processing node:", nodeId)
        lines.append(f"[{nodeId}] {title}\n.  Summary: {summary}")
    return "\n".join(lines)

def call_llm(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    llm = ChatGroq(
        model=model,
        api_key=GROQ_API_KEY,
        temperature=0,
    )
    response = llm.invoke(prompt)
    return response.content


def custom_pipeline(docId: str, question: str) -> str:
    if docId is None:
        raise ValueError("docId must be provided.")
    
    # Step 1: pull the hierarchical tree (titles + summaries + node ids)
    treeResult=pageIndexClient.get_tree(docId)
    tree=treeResult["result"]
    flatNodes = flattenTree(tree)
    outline = buildSummaryOutline(flatNodes)

    # Step 2: ask the LLM which node(s) are likely to contain the answer,
    # using ONLY titles/summaries (cheap, no full text yet)
    selection_prompt = f"""You are given a question and a table-of-contents-style
outline of a document, where each entry has a node id, a title, and a summary.

Question: {question}

Outline:
{outline}

Return ONLY a JSON list of the node_id values most likely to contain the
answer, e.g. ["0003", "0007"]. Pick as few as needed, but don't miss relevant ones."""
    
    rawSelectedIds = call_llm(selection_prompt)
    try:
        selected_ids = json.loads(rawSelectedIds)
    except json.JSONDecodeError:
        # model may wrap in prose/backticks despite instructions; do a light cleanup
        cleaned = rawSelectedIds.strip().strip("`").strip()
        selected_ids = json.loads(cleaned)

    idNodeMapping = {(n.get("node_id") or n.get("id")): n for n in flatNodes}

    selectedTexts = []
    for nodeId in selected_ids:
        node=idNodeMapping.get(nodeId)
        if node:
            text = node.get("text") or node.get("content") or ""
            selectedTexts.append(f"### {node.get("title", nodeId)}\n{text}")

    context = "\n\n".join(selectedTexts)

    # Step 4: generate the grounded answer using only the retrieved context
    answer_prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say so.

Context:
{context}

Question: {question}"""
    
    return call_llm(answer_prompt)

if __name__ == "__main__":
    docId = submitAndWait(DOC_PATH)

    question = "What is the main topic of the document?"

    # print("\n--- Mode A: built-in PageIndex chat_completions ---")
    # print(useBuiltInChat(docId, question))

    print("\n--- Mode B: custom tree-reasoning pipeline ---")
    print(custom_pipeline(docId, question))