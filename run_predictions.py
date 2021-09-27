{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2e58f234",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Initial log joint probability = -5.49514\n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "      95       1630.28    0.00351769       300.562   3.808e-05       0.001      149  LS failed, Hessian reset \n",
      "      99       1630.44    0.00121807       122.958      0.6663      0.6663      154   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     148       1630.58   9.23795e-08       57.3011      0.2376           1      216   \n",
      "Optimization terminated normally: \n",
      "  Convergence detected: relative gradient magnitude is below tolerance\n",
      "Initial log joint probability = -6.48249\n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "      99       2787.12    0.00186413       112.664       0.565       0.565      131   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     118       2787.72   0.000595684       326.117   6.392e-06       0.001      191  LS failed, Hessian reset \n",
      "     199       2789.56    0.00363797       132.246           1           1      292   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     233       2790.52   8.43628e-05       87.2833   5.075e-07       0.001      387  LS failed, Hessian reset \n",
      "     299       2791.13   0.000257279       81.0769           1           1      475   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     327       2791.42   7.73526e-05       97.7767   6.852e-07       0.001      554  LS failed, Hessian reset \n",
      "     399       2791.61   4.24948e-06       71.0549      0.9765      0.9765      643   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     426       2791.61   2.81159e-05        84.146   3.301e-07       0.001      732  LS failed, Hessian reset \n",
      "     499       2791.66    1.3929e-05       89.7471      0.7729      0.7729      824   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     550       2791.88   0.000769784       94.8762   4.513e-06       0.001      933  LS failed, Hessian reset \n",
      "     599       2792.08   8.07814e-05       45.3275           1           1      999   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     625       2792.11   9.97356e-05       83.4748   1.237e-06       0.001     1067  LS failed, Hessian reset \n",
      "     699       2792.13   8.60836e-05       65.6816      0.4185           1     1152   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     799       2793.18    0.00217046       145.524           1           1     1267   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     801       2793.25   0.000118077       148.598   8.683e-07       0.001     1305  LS failed, Hessian reset \n",
      "     899       2794.03   3.13802e-06       63.8678           1           1     1425   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     978       2794.06    7.0084e-07        72.991       0.766       0.766     1522   \n",
      "Optimization terminated normally: \n",
      "  Convergence detected: relative gradient magnitude is below tolerance\n",
      "Initial log joint probability = -30.8616\n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "      99       1945.34    0.00688935       470.974      0.4067           1      137   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     199       1953.57     0.0423117       107.259      0.4179           1      266   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     299       1959.79   0.000219593       68.2836           1           1      399   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     341       1960.95    0.00344491       245.222   3.518e-05       0.001      486  LS failed, Hessian reset \n",
      "     399       1961.23   3.93672e-06       63.1709      0.3219      0.3219      567   \n",
      "    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes \n",
      "     432        1961.3    0.00020941       75.8211   2.673e-06       0.001      657  LS failed, Hessian reset \n",
      "     454       1961.32   1.61242e-07        57.087      0.3526           1      694   \n",
      "Optimization terminated normally: \n",
      "  Convergence detected: relative gradient magnitude is below tolerance\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from prophet import Prophet\n",
    "from prophet.plot import plot_plotly, plot_components_plotly\n",
    "\n",
    "# Python\n",
    "df = pd.read_csv('data/daily_data.csv')\n",
    "\n",
    "for company_id in df.company_id.unique():\n",
    "    df_company = df[df.company_id == company_id]\n",
    "    df_prophet = df_company[['date', 'revenue']].copy()\n",
    "    df_prophet = df_prophet.rename(columns={\"date\": \"ds\", \"revenue\": \"y\"})\n",
    "\n",
    "    m = Prophet(daily_seasonality=True)\n",
    "    m.fit(df_prophet)\n",
    "\n",
    "    future = m.make_future_dataframe(periods=150)\n",
    "    forecast = m.predict(future)\n",
    "\n",
    "    plot_plotly(m, forecast)\n",
    "    plot_components_plotly(m, forecast)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ffe7c686",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
