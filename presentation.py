import marimo

__generated_with = "0.13.15"
app = marimo.App(
    width="medium",
    layout_file="layouts/presentation.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import mosqlient as mq
    import datetime as dt
    import pandas as pd
    import altair as alt

    return alt, dt, mo, mq, pd


@app.cell
def _(mo):
    mo.md(
        r"""
    # Dengue Forecasting
    ### Flávio Codeço Coelho
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## The challenge""")
    return


@app.cell
def _(mo):
    import os
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Access the environment variables
    api_key = os.getenv('MOSQLIMATE_API_KEY')
    ufs = {
        "Acre": "AC",
        "Alagoas": "AL",
        "Amazonas": "AM",
        "Amapá": "AP",
        "Bahia": "BA",
        "Ceará": "CE",
        "Distrito Federal": "DF",
        "Espírito Santo": "ES",
        "Goiás": "GO",
        "Maranhão": "MA",
        "Minas Gerais": "MG",
        "Mato Grosso do Sul": "MS",
        "Mato Grosso": "MT",
        "Pará": "PA",
        "Paraíba": "PB",
        "Pernambuco": "PE",
        "Piauí": "PI",
        "Paraná": "PR",
        "Rio de Janeiro": "RJ",
        "Rio Grande do Norte": "RN",
        "Roraima": "RR",
        "Rio Grande do Sul": "RS",
        "Rondonia": "RO",
        "Tocantins": "TO"
    }
    state = mo.ui.dropdown(options=list(ufs.keys()), value='Distrito Federal')
    return api_key, state, ufs


@app.cell
def _(api_key, mo, mq, pd, ufs):
    @mo.cache
    def fetch_dengue_data(start, stop, uf):
        df = mq.get_infodengue(
            api_key=api_key,
            disease="dengue",
            uf=ufs[uf],
            start_date=start,
            end_date=stop,
        )
        #select columns and rename the `casprov` column.
        print(df.columns)
        columns = ['data_iniSE', 'SE', 'municipio_geocodigo', 'municipio_nome', 'casprov', 'Rt', 'p_rt1', 'nivel']
        df = df[columns].rename(columns = {'casprov':'casos', 'data_iniSE':'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by = 'date')
        return df
    return (fetch_dengue_data,)


@app.cell
def _(dt, mo):
    start = mo.ui.date(value='2024-01-01', start='2010-01-01', stop=dt.date.today())
    stop = mo.ui.date(value='2025-01-01', start='2010-01-02', stop=dt.date.today())



    return start, stop


@app.cell
def _(alt):
    def gen_tsplot(df, city='all'):
        if city == "all":
            df  = df.groupby('date').agg({'casos':'sum'})
        else:
            df = df[df.municipio_nome==city]
        tooltips = ["date:T", "casos:Q", "nivel:N"]
        line = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(x="date:T", y="casos:Q", tooltip=tooltips)
        )

        # Criando áreas coloridas por nível
        areas = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3)
            .encode(
                x="date:T",
                y="casos:Q",

                color=alt.Color(
                    field="nivel",
                    scale={
                        'domain':[1,2,3,4],
                        'range':['green','yellow','orange','red']
                    },
                ),
            )
        )

        # Combinando os gráficos
        plot = (line+areas).properties(
            width=800,
            height=400,
            title=f"Casos de dengue ao longo do tempo por nível de alerta em {city}",
        )
        plot = plot.configure_mark(
            tooltip=alt.TooltipContent(
                content="encoding"  # Mostra todos os campos codificados
            )
        )
        return plot
    return (gen_tsplot,)


@app.cell
def _(fetch_dengue_data, start, state, stop):
    dengue = fetch_dengue_data(start.value, stop.value, state.value)
    return (dengue,)


@app.cell
def _(dengue, mo):
    lista = list(set(dengue.municipio_nome.astype(str)))
    cities = mo.ui.dropdown(options=lista, value=lista[0])
    state_wide = mo.ui.checkbox(value=False)
    return cities, state_wide


@app.cell
def _(cities, dengue, gen_tsplot, mo, start, state, state_wide, stop):
    cn  = cities.value
    if state_wide.value:
        cn="all"
    tsp = mo.ui.altair_chart(gen_tsplot(dengue, cn))
    mo.md(
    f"""
    ## Selecting our dataset:
    **Start date:** {start} **End date:** {stop} **State:** {state} **City:** {cities} *plot state series:* {state_wide}

    {tsp}
    """
    )

    return


@app.cell
def _(alt, dengue):
    alt.Chart(dengue).mark_bar(
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2
    ).encode(
        x='date:T',
        y='casos:Q',
        color=alt.Color(
                    field="nivel",
                    scale={
                        'domain':[1,2,3,4],
                        'range':['green','yellow','orange','red']
                    },
                    title="Alerta"
                ),
    )

    return


@app.cell
def _(dengue):
    dengue.info()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
