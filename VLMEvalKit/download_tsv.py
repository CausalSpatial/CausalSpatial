import pandas as pd


from huggingface_hub import hf_hub_download


repo_id = "Mwxinnn/CausalSpatial"
filename = "VLMEvalKit/CausalSpatial.tsv" 


file_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset"
)

df = pd.read_csv(file_path, sep='\t')
print(df.head())