# Dengue Data Analysis Dashboard

Dashboard interativo para análise e previsão de casos de dengue no Brasil utilizando marimo.

## Pré-requisitos

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) (recomendado) ou pip
- [marimo](https://marimo.io/) instalado globalmente

## Configuração

1. Primeiro, crie e ative um ambiente virtual com UV:

```bash
uv venv .venv
source .venv/bin/activate.fish
```

2. Instale as dependências:

```bash
uv pip install -r requirements.txt
```

3. Crie um arquivo `.env` com sua chave de API:

```bash
echo "MOSQLIMATE_API_KEY=sua_chave_aqui" > .env
```

4. Execute o dashboard:

```bash
marimo run presentation.py
```

## Funcionalidades

- Visualização de séries temporais por município/estado
- Análise de sazonalidade com gráficos polares
- Denoising de séries temporais usando wavelets
- Modelagem e previsão com Local Linear Trend
- Visualização interativa com filtros por:
  - Período temporal
  - Estado brasileiro
  - Município específico

## Dados

Os dados são obtidos através da API do [InfoDengue/MoSquimate](https://info.dengue.mat.br/).

## Desenvolvimento

Para contribuir com o projeto:

1. Instale as dependências de desenvolvimento:

```bash
uv pip install -r requirements-dev.txt
```

2. Execute os testes:

```bash
pytest
```

## Licença

[MIT](LICENSE)
