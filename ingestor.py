from unstructured.partition.pdf import partition_pdf

from unstructured.documents.elements import NarrativeText,Title,ListItem

from langchain_core.documents import Document

allowed_types=(NarrativeText,Title,ListItem)

def load_pdf(file):
    elements=partition_pdf(file,strategy="fast")

    filtered_docs=[]
    for e in elements:
        if isinstance(e,allowed_types):
            text=str(e).strip()
            if len(text)>40:
                filtered_docs.append(text)
    docs=[Document(page_content=d,metadata={"name":str(file)}) for d in filtered_docs]
    return docs