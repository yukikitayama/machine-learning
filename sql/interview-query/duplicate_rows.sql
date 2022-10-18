with cte as (
  select
    *,
    row_number() over(partition by id order by created_at) as row_num
  from
    users
)

select
  id,
  name,
  created_at
from
  cte
where
  row_num != 1
;