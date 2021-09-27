import pandas as pd
import numpy as np
from prophet import Prophet

# Python
df = pd.read_csv('data/daily_data.csv')

forecast_period = 150
floor = 0


class ProphetPos(Prophet):
    @staticmethod
    def piecewise_linear(t, deltas, k, m, changepoint_ts):
        """Evaluate the piecewise linear function, keeping the trend
        positive.

        Parameters
        ----------
        t: np.array of times on which the function is evaluated.
        deltas: np.array of rate changes at each changepoint.
        k: Float initial rate.
        m: Float initial offset.
        changepoint_ts: np.array of changepoint times.

        Returns
        -------
        Vector trend(t).
        """
        # Intercept changes
        gammas = -changepoint_ts * deltas
        # Get cumulative slope and intercept at each t
        k_t = k * np.ones_like(t)
        m_t = m * np.ones_like(t)
        for s, t_s in enumerate(changepoint_ts):
            indx = t >= t_s
            k_t[indx] += deltas[s]
            m_t[indx] += gammas[s]
        trend = k_t * t + m_t
        if max(t) <= 1:
            return trend
        # Add additional deltas to force future trend to be positive
        indx_future = np.argmax(t >= 1)
        while min(trend[indx_future:]) < 0:
            indx_neg = indx_future + np.argmax(trend[indx_future:] < 0)
            k_t[indx_neg:] -= k_t[indx_neg]
            m_t[indx_neg:] -= m_t[indx_neg]
            trend = k_t * t + m_t
        return trend

    def predict(self, df=None):
        fcst = super().predict(df=df)
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            fcst[col] = fcst[col].clip(lower=0.0)
        return fcst

for company_id in df.company_id.unique():
    df_company = df[df.company_id == company_id]
    df_prophet = df_company[['date', 'revenue']].copy()
    df_prophet = df_prophet.rename(columns={"date": "ds", "revenue": "y"})

    m = ProphetPos(daily_seasonality=False)
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=forecast_period)
    forecast = m.predict(future)

    samples = m.predictive_samples(future)
    yhats   = samples['yhat']
    yhats[yhats<floor] = floor
    predictive_sum_samples = yhats[-forecast_period:].sum(axis=0)

    total_forecasted_revenue = '${:,.2f}'.format(predictive_sum_samples.mean())
    upper_forecasted_revenue = '${:,.2f}'.format(np.percentile(predictive_sum_samples, 90))
    lower_forecasted_revenue = '${:,.2f}'.format(np.percentile(predictive_sum_samples, 10))

    fig1 = m.plot(forecast, xlabel='Date', ylabel='Daily Revenue')
    fig2 = m.plot_components(forecast)

    fig1.savefig(f'figures/{company_id}_revenue_forecast.png')
    fig2.savefig(f'figures/{company_id}_seasonality_analysis.png')

    print(f'{company_id} forecasted revenue for next {forecast_period} days:' \
           f' {total_forecasted_revenue} ({lower_forecasted_revenue}, {upper_forecasted_revenue}) 80% confidence interval')











