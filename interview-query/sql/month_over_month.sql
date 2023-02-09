-- Compute monthly revenue
with cte as (
  select
    month(a.created_at) as "month",
    sum(a.quantity * b.price) as revenue
  from
    transactions as a
  left join
    products as b
  on
    a.product_id = b.id
  where
    year(a.created_at) = 2019
  group by
    1
)

select
  a.month,
  round((a.revenue - b.revenue) / b.revenue, 2) as month_over_month
from
  cte as a
left join
  cte as b
on
  a.month = b.month + 1
;
