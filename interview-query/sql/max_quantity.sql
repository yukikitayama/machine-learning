select
  year(created_at) as year,
  product_id,
  max(quantity) as max_quantity
from
  transactions
group by
  1,
  2
order by
  1,
  2
;
