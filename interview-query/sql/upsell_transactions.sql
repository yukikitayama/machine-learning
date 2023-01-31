with cte as (
  select
    distinct user_id
  from
    transactions
  group by
    user_id
  having
    count(distinct date(created_at)) > 1
)

select
  count(user_id) as num_of_upsold_customers
from
  cte
;
