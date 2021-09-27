# Data Analyst Project
You have been provided the ecommerce order history for three different companies. Each row
provides information on:
* order_id: a unique identifier for the order 
* company_id: the Company name
* created_at: the date on which the order was created
* status: the status of the order
* total_price: the total order value
* order_refunds: the total that got refunded if at all
* customer_id: a unique identifier of the customer who make the purchase
* customer_order_count: the total number of orders placed by the customer by the time
the order was created, including the current one
* customer_total_spent: the total amount spent by the customer by the time the order was
created, including the current one
* customer_created_at: and the date on which the customer was created / registered


The risk team is interested in building out a dashboard to assist in the underwriting of these 
ecommerce businesses.
1. Please use the data to build out a dashboard that can be used for all companies with the
same data structures.
2. Once the dashboard is complete, please put together a brief summary of your findings
and observations on these three companies. Please be succinct in your response.
3. Bonus Project: The risk team is also interested in projecting each company's total
ecommerce revenue over the next 150 days. From a risk perspective, it is helpful to
know what the projected number is over the 150-day period, and it is also helpful to
know the range and likelihood of projected outcomes. Construct a model that you
believe best assists in answering this question.

#### Requirements and Hints:
* For purposes of this exercise, you can assume revenue to be total_price net of
order_refunds for all paid orders.
* Please commit your solution in a separate branch and create a Merge Request with your
changes.
* Please write your code in python and your written response in Word. You are welcome
to use any open-source packages / libraries in the python ecosystem.
* Please provide some minimal documentation on how to run your model.

#### Run instructions:
* app.py runs the dashboard server
* run_revenue_predictions.py refreshes the figures in assets and runs the prediction model
* create_daily_data.py resets the daily_data.csv cache if there are changes to customer_data.cvs (should not need to run this)
