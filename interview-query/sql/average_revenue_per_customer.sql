-- Average revenue per client = total revenue / number of clients
-- revenue = amount_per_unit * quantity
-- total revenue = sum of revenue
select
  round(
    sum(amount_per_unit * quantity)
    /
    count(distinct user_id)
  , 2) as average_lifetime_revenue
from
  payments