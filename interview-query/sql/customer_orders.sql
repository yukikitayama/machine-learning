-- user_id who placed more than 3 transactions in 2019
with
cte1 as (
select
  user_id
from
  transactions
where
  year(created_at) = 2019
group by
  user_id
having
  count(*) > 3
),

cte2 as (
select
  user_id
from
  transactions
where
  year(created_at) = 2020
group by
  user_id
having
  count(*) > 3
)

select
  name as customer_name
from
  users
where
  id in (select * from cte1)
  and id in (select * from cte2)
;
