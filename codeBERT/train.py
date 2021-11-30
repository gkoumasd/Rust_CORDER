import torch
from transformers import RobertaConfig, RobertaModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RobertaModel.from_pretrained("microsoft/codebert-base")