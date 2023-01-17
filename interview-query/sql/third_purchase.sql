with cte as (
  select
    user_id,
    created_at,
    product_id,
    quantity,
    row_number() over(
      partition by user_id
      order by created_at, id
    ) as row_num
  from
    transactions
)

select
  user_id,
  created_at,
  product_id,
  quantity
from
  cte
where
  row_num = 3
;