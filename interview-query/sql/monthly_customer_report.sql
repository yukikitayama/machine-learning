select
  month(a.created_at) as month,
  count(distinct a.user_id) as num_customers,
  count(distinct a.id) as num_orders,
  sum(a.quantity * b.price) as order_amt
from
  transactions as a
left join
  products as b
on
  a.product_id = b.id
where
  year(a.created_at) = 2020
group by
  1
;
