"""
Interactive presentation for dengue fever data analysis and forecasting.

Features:
- Data visualization of dengue cases by state/city
- Seasonal pattern analysis
- Time series denoising using wavelets
- Forecasting with local linear trend models

Requires:
- marimo
- mosqlient
- pandas
- numpy
- altair
- matplotlib
- pywavelets
- statsmodels
"""

import marimo

__generated_with = "0.14.7"
app = marimo.App(
    width="medium",
    layout_file="layouts/presentation.slides.json",
)


@app.cell
def _():
    # Core libraries
    import marimo as mo
    import datetime as dt

    # Data handling
    import pandas as pd
    import numpy as np

    # Visualization
    import altair as alt
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm, colors

    # Time series analysis
    import pywt
    import statsmodels.api as sm
    from scipy.stats import norm

    # Data API
    import mosqlient as mq

    return alt, cm, colors, dt, mo, mq, np, pd, plt, pywt, sm


@app.cell
def _(mo):
    mo.md(
        r"""
    # Dengue Forecasting Dashboard
    ### Análise e previsão de casos de dengue no Brasil

    **Autor:** Flávio Codeço Coelho  
    **Data:** 25/06/2025

    Este dashboard interativo permite:

    - Visualizar séries temporais de casos de dengue
    - Analisar padrões sazonais
    - Aplicar técnicas de denoising
    - Realizar previsões de curto prazo
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
        """
        Fetch dengue case data from InfoDengue API.

        Args:
            start (str): Start date in YYYY-MM-DD format
            stop (str): End date in YYYY-MM-DD format 
            uf (str): Brazilian state name

        Returns:
            pd.DataFrame: DataFrame with dengue cases data
        """
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
def _():
    # alt.Chart(dengue).mark_bar(
    #     cornerRadiusTopLeft=2,
    #     cornerRadiusTopRight=2
    # ).encode(
    #     x='date:T',
    #     y='casos:Q',
    #     color=alt.Color(
    #                 field="nivel",
    #                 scale={
    #                     'domain':[1,2,3,4],
    #                     'range':['green','yellow','orange','red']
    #                 },
    #                 title="Alerta"
    #             ),
    # )

    return


@app.cell
def _():
    ## Seasonality
    return


@app.cell
def _(cm, colors, np, plt):
    def plot_polar(df, disease, city):
        if city != 'all':
            df = df[df.municipio_nome==city]
        df.loc[:,'EW'] = [int(str(s)[-2:]) for s in df.SE]
        df.loc[:,'year'] = [int(str(s)[:-2]) for s in df.SE]
        df2 = df.sort_values('date')
        df2['ew_r'] = df2.EW*(2*np.pi/52.)
        cmap = cm.jet((df2.year-df.year.min())/(df2.year.max()-df.year.min()))
        fig, [ax1, ax2] = plt.subplots(1,2,subplot_kw={'projection': 'polar'}, figsize=(13,6))
        ax1.set_xticklabels(np.linspace(1,52,9, dtype=int))
        ax1.set_title(f'Casos de {disease} entre 2010 e 2025 - {city}')
        ax2.set_xticklabels(np.linspace(1,52,9, dtype=int))
        ax2.set_yscale("log")
        ax2.set_title(f'Log(Casos) de {disease} entre 2010 e 2025 - {city}')

        ax1.plot(df2.ew_r,df2.casos,color='gray',lw=1, alpha=0.3)
        ax2.plot(df2.ew_r,df2.casos,color='gray',lw=1, alpha=0.3)
        sct1 = ax1.scatter(df2.ew_r,df2.casos,s=10, c=cmap);
        sct2 = ax2.scatter(df2.ew_r,df2.casos,s=10, c=cmap);
        # cb = plt.colorbar(sct, ax=ax)
        cb2 = fig.colorbar(cm.ScalarMappable(colors.Normalize(df2.year.min(),df2.year.max(), True), cmap=cm.jet), ax=ax2)

        cb2.set_ticklabels([str(y) for y in range(df2.year.min(), df2.year.max()+1)])
        cb2.set_ticks(ticks=np.linspace(0,1, df2.year.max()-df2.year.min()+1), labels=[str(y) for y in range(df2.year.min(), df2.year.max()+1)])
        # cb2 = fig.colorbar(cm.ScalarMappable(colors.Normalize(2010,2026, True), cmap=cm.jet), ax=ax2)
        # cb2.set_ticks(ticks=np.linspace(0,1, 16), labels=[str(y) for y in range(df2.year.min(), df2.year.max()+1)])


        ax1.set_xlabel('Semanas');
        ax2.set_xlabel('Semanas')
        return fig
    return (plot_polar,)


@app.cell
def _(cities, dengue, plot_polar):
    plot_polar(dengue,'dengue', cities.value)
    return


@app.cell
def _():
    ## Preprocessing
    return


@app.cell
def _(dengue, mo, pywt):

    wvlt='coif4'
    coeffs = pywt.wavedec(dengue.casos/dengue.casos.max(), wvlt)
    # Set a threshold to nullify smaller coefficients (assumed to be noise)
    threshold = mo.ui.slider(start=0.1, stop=1, step=0.001, value=0.211)

    return coeffs, threshold, wvlt


@app.cell
def _(alt):
    def plot_denoised(signal, denoised_signal
                     ):
        signal['denoised'] = denoised_signal[:len(signal)]
        line = (
            alt.Chart(signal)
            .mark_line(point=True)
            .encode(x="date:T", y="casos:Q")
        ).properties(title="Noisy Series")
        line2 = (
            alt.Chart(signal)
            .mark_line(point=True)
            .encode(x="date:T", y="denoised:Q")
        ).properties(title="Denoised Series", color='green')


        return line|line2
    return (plot_denoised,)


@app.cell
def _(coeffs, dengue, mo, plot_denoised, pywt, threshold, wvlt):

    coeffs_thresholded = [pywt.threshold(c, threshold.value, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(coeffs_thresholded, wvlt)*dengue.casos.max()
    mo.md(
    f'''
    ### Denoising  the series
    **Threshold:** {threshold} {threshold.value}

    {mo.ui.altair_chart(plot_denoised(dengue, denoised_signal))}
    '''
    )

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Simple Forecast
    To exemplify how we can approach the forecast of the dengue inference time series, we will fit a state-space model to it.
    """
    )
    return


@app.cell
def _(np, sm):
    class LocalLinearTrend(sm.tsa.statespace.MLEModel):
        def __init__(self, endog):
            # Model order
            k_states = k_posdef = 2

            # Initialize the statespace
            super(LocalLinearTrend, self).__init__(
                endog,
                k_states=k_states,
                k_posdef=k_posdef,
                initialization="approximate_diffuse",
                loglikelihood_burn=k_states,
            )

            # Initialize the matrices
            self.ssm["design"] = np.array([1, 0])
            self.ssm["transition"] = np.array([[1, 1], [0, 1]])
            self.ssm["selection"] = np.eye(k_states)

            # Cache some indices
            self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

        @property
        def param_names(self):
            return ["sigma2.measurement", "sigma2.level", "sigma2.trend"]

        @property
        def start_params(self):
            return [np.std(self.endog)] * 3

        def transform_params(self, unconstrained):
            return unconstrained**2

        def untransform_params(self, constrained):
            return constrained**0.5

        def update(self, params, *args, **kwargs):
            params = super(LocalLinearTrend, self).update(params, *args, **kwargs)

            # Observation covariance
            self.ssm["obs_cov", 0, 0] = params[0]

            # State covariance
            self.ssm[self._state_cov_idx] = params[1:]
    return (LocalLinearTrend,)


@app.cell
def _(LocalLinearTrend, dengue, mo, np):
    dengue['log_casos'] = np.log(dengue.casos)
    # Setup the model
    mod = LocalLinearTrend(dengue.set_index('date')["log_casos"])

    # Fit it using MLE (recall that we are fitting the three variance parameters)
    res = mod.fit(disp=False)
    # print(res.summary())
    mo.md(f'''
    When we run this regression model, we get:

    ```
    {res.summary()}
    ```
    ''')
    return (res,)


@app.cell
def _(res):
    # Perform prediction and forecasting
    predict = res.get_prediction()
    forecast = res.get_forecast(20)

    return forecast, predict


@app.cell
def _():
    # dengue
    return


@app.cell
def _(dengue, forecast, np, plt, predict):
    def plot_forecast():
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot the results
        dengue.set_index('date')["log_casos"].plot(ax=ax, style="k.", label="Observations")
        predict.predicted_mean.plot(ax=ax, label="One-step-ahead Prediction")
        predict_ci = predict.conf_int(alpha=0.05)
        predict_index = np.arange(len(predict_ci))
        ax.fill_between(
            predict_index[2:], predict_ci.iloc[2:, 0], predict_ci.iloc[2:, 1], alpha=0.1
        )

        forecast.predicted_mean.plot(ax=ax, style="r", label="Forecast")
        forecast_ci = forecast.conf_int()
        forecast_index = np.arange(len(predict_ci), len(predict_ci) + len(forecast_ci))
        ax.fill_between(
            forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], alpha=0.1
        )

        # Cleanup the image
        ax.set_ylim((4, 8))
        legend = ax.legend(loc="lower left");
        return fig
    return (plot_forecast,)


@app.cell
def _(plot_forecast):
    plot_forecast()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Covariates and Causal Relations
    Now let's explore the relationship of incidence with climate covariates
    """
    )
    return


@app.cell
def _():
    from ftplib import FTP
    return


@app.cell
def _():
    ftp_host = 'info.dengue.mat.br'
    ftp_user = 'anonymous'
    # create a connection with this FTP host and download the file 'data_sprint_2025/climate_csv_gz'. AI!
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
