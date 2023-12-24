-- https://www.interviewquery.com/questions/weighted-average-sales

select
  a.date,
  a.product_id,
  0.5 * a.sales_volume + 0.3 * b.sales_volume + 0.2 * c.sales_volume as weighted_avg_sales
from
  sales as a
inner join
  sales as b
on
  datediff(a.date, b.date) = 1
  and a.product_id = b.product_id
inner join
  sales as c
on
  datediff(a.date, c.date) = 2
  and a.product_id = c.product_id
order by
  1,
  2
;