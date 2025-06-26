# Dengue Data Analysis Dashboard

Dashboard interativo para análise e previsão de casos de dengue no Brasil utilizando marimo.

## Pré-requisitos

- Python 3.12+


## Configuração

1. Primeiro, crie e ative um ambiente virtual com UV e instale as dependências:

```bash
uv sync
source .venv/bin/activate.fish
```
or, if you use bash just `source .venv/bin/activate`

2. Crie um arquivo `.env` com sua chave de API:

```bash
echo "MOSQLIMATE_API_KEY=sua_chave_aqui" > .env
```

4. Execute o dashboard:

```bash
marimo run presentation.py
```

Ou para editar:

```
marimo edit presentation.py
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

Os dados são obtidos através da API do [Mosquimate](https://api.mosqlimate.org/).


## Licença

[MIT](LICENSE)
