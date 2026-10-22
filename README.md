# Kaggle + GitHub Template

Template pronto para clonar no [Kaggle](https://www.kaggle.com/code) e rodar projetos do GitHub com GPU/TPU.

## Estrutura

```
kaggle-template/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── train.py
├── predict.py
└── notebooks/
    └── kaggle_template.ipynb
```

## Como usar no Kaggle

1. Crie um **New Notebook** no Kaggle.
2. Em **Settings**, ative **GPU** ou **TPU** (se for deep learning).
3. Na aba **Add-ons**, conecte sua conta do GitHub.
4. No notebook, rode a célula de clonagem:

```python
!git clone https://github.com/SEU_USUARIO/kaggle-template.git /kaggle/working/kaggle-template
!pip install -r /kaggle/working/kaggle-template/requirements.txt -q
```

5. Importe e execute os scripts:

```python
import sys
sys.path.insert(0, "/kaggle/working/kaggle-template")

from train import train
from predict import predict
```

Ou abra o notebook `notebooks/kaggle_template.ipynb` e execute célula por célula.

## Secrets do Kaggle (repositórios privados)

Se o repositório for privado, adicione em **Settings → Secrets**:
- `GITHUB_TOKEN`: seu Personal Access Token do GitHub

Luego use:
```python
!git clone https://{GITHUB_TOKEN}@github.com/SEU_USUARIO/kaggle-template.git
```

## Licença

MIT