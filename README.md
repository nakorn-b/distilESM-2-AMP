<h1 align="center">🧬 distilESM-2-AMP</h1>

distilESM-2-AMP is a  distilled version of META AI's `esm2_t6_8M_UR50D`. The research focuses on model compression and classification of Antimicrobial Peptide (AMP) in bioinformatics field.
### Paper & Resources


<!-- - Full Paper: [distilESM-2-AMP: Effective LLM Distillation for AMPs]() -->

- Model on Hugging Face: [NakornB/distilESM-2-AMP](https://huggingface.co/NakornB/distilESM-2-AMP)

- Live Demo: Check out [webapp]([https://distilesm-2-amp.streamlit.app/](https://distil-esm-2-amp.vercel.app/)) here

## 📊 Model comparision
|Name|Parameter| Pretraining data | Protein sequence | Size (MB) |
|:---:|:---:| :---: | :-:| :-:|
|esm2_t6_8M_UR50D| 8M |UR50D | 65M | 7.51M |
|distilESM-2-AMP| 4M| UniRef50 | 8M | 3.81M |

## ⚒️ Installation & Usage
```bash
pip install -U transformers torch
```
```
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "NakornB/distilESM-2-AMP"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### Contact
- Email: boonprasongnakorn@gmail.com
- GitHub: [nemo-netidol](https://github.com/Nemo-netidol)
- Linkedln: https://www.linkedin.com/in/nakornb/

### License 
[MIT License](LICENSE)
