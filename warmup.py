from unstructured.partition.pdf import partition_pdf

# trigger model download
partition_pdf(
    filename="attention.pdf",
    strategy="hi_res",
    infer_table_structure=True
)

print("PDF Model downloaded successfully")