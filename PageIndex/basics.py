import os
import time

from pageindex import PageIndexClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
PAGEINDEX_API_KEY=os.getenv("PAGE_INDEX_API_KEY")
DOC_PATH = "./r1200appendix_d.pdf"

page_index_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

def submit_and_wait(doc_path:str|None=None, poll_seconds=5):
    if doc_path is None:
        raise ValueError("doc_path must be provided.")
    result = page_index_client.submit_document(doc_path)
    doc_id = result["doc_id"]

    while True:
        status = page_index_client.get_document(doc_id)["status"]

        if status=="completed":
            print("PageIndex, DocId-:", doc_id)
            print("Document Processing Completed")
            print("Result",result)

            break
        elif status=="failed":
            print("Document Processing Failed")
            break
        else:
            print("Still processing... checking again in", poll_seconds, "s")
        time.sleep(poll_seconds)
    return doc_id

if __name__ == "__main__":
    doc_id = submit_and_wait(DOC_PATH)

    print("--------------Visualizing the tree structure of the documnet---------------")
    document_tree = page_index_client.get_tree(doc_id)
    print(document_tree)
    print("--------------Visualizing the tree structure of the documnet---------------")
