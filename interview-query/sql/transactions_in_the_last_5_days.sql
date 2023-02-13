with

-- Number of transactions for each user and each date
cte1 as (
  select
    user_id,
    date(created_at) as created_date,
    count(*) as transaction_count
  from
    bank_transactions
  where
    date(created_at) between "2020-01-01" and "2020-01-05"
  group by
    1,
    2
),

-- Users who made at least one transactions each day
cte2 as (
  select
    user_id
  from
    cte1
  group by
    user_id
  having
    count(*) = 5
)

select
  count(user_id) as number_of_users
from
  cte2
;