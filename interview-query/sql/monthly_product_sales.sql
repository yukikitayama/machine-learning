select
  left(month, 10) as "month",
  sum(if(product_id = 1, amount_sold, 0)) as "1",
  sum(if(product_id = 2, amount_sold, 0)) as "2",
  sum(if(product_id = 3, amount_sold, 0)) as "3",
  sum(if(product_id = 4, amount_sold, 0)) as "4"
from
  monthly_sales
group by
  1
;