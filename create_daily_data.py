import pandas as pd
import numpy as np


order_data = pd.read_csv("data/customer_data.csv")


datetimes = pd.to_datetime(order_data["created_at"])
order_data["date"] = datetimes.dt.date
order_data["month"] = datetimes.dt.strftime('%b')
order_data["revenue"] = order_data['total_price'] - order_data['order_refunds']
order_data["customer_lifespan"] = pd.to_timedelta(pd.to_datetime(order_data['created_at']) - pd.to_datetime(order_data['customer_created_at']))
order_data.sort_values("date", inplace=True)

daily_data = pd.DataFrame(columns=['date', 'company_id', 'revenue', 'num_orders', 'revenue_per_order', 'average_customer_lifespan', 'number_of_unique_customers', 'month'])

grouped = order_data.groupby(['date', 'company_id'])

customer_data = dict()
daily_row_at_add = dict()
for name, group in grouped:

    company_id = name[1]
    if company_id not in customer_data:
        customer_data[company_id] = dict()

    daily_row_at_add['date'] = name[0]
    daily_row_at_add['company_id'] = company_id
    daily_row_at_add['revenue'] = group.revenue.sum()
    daily_row_at_add['num_orders'] = group.shape[0]
    daily_row_at_add['revenue_per_order'] = group.revenue.sum() / group.shape[0]

    new_customers_bool = ~(group.customer_id.isin(customer_data[company_id].keys()))
    new_customers = new_customers_bool.sum()

    daily_row_at_add['new_customers'] = new_customers
    daily_row_at_add['month'] = group.month.unique()[0]

    for index, row in group.iterrows():
        customer_id = row.customer_id
        if customer_id not in customer_data[company_id]:
            customer_data[company_id][customer_id] = (0, 0, pd.Timedelta("0 days"))
        current_customer_orders, current_customer_revenue, _ = customer_data[company_id][customer_id]
        customer_data[company_id][customer_id] = (current_customer_orders+1, current_customer_revenue+row.revenue, row.customer_lifespan)

    company_customer_data = customer_data[company_id]
    orders = pd.DataFrame([data[0] for data in company_customer_data.values()])
    revenue = pd.DataFrame([data[1] for data in company_customer_data.values()])
    lifespans = pd.DataFrame([data[2] for data in company_customer_data.values()])
    lifespans_years = lifespans / np.timedelta64(1, 'Y')

    num_customers = len(company_customer_data.keys())
    num_orders = orders.sum()
    average_purchase_value = revenue.sum() / num_orders
    average_purchase_frequency_rate = num_orders / num_customers
    customer_value = average_purchase_value * average_purchase_frequency_rate
    average_customer_lifespan = lifespans_years.mean()

    daily_row_at_add['repeat_customer_rate'] = float(len(['' for data in company_customer_data.values() if data[0] > 1]) / num_customers)
    daily_row_at_add['churn_rate'] = float(num_customers / num_orders)
    daily_row_at_add['customer_lifetime_value'] = float(average_customer_lifespan * customer_value)

    daily_data = daily_data.append(daily_row_at_add, ignore_index=True)

daily_data = pd.get_dummies(daily_data,prefix=['month'], columns = ['month'], drop_first=False)

daily_data.to_csv('data/daily_data.csv', index=False)


