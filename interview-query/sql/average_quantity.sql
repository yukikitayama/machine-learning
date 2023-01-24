select
  year(created_at) as "year",
  product_id,
  round(avg(quantity), 2) as avg_quantity
from
  transactions
group by
  1,
  2
order by
  1,
  2
;
