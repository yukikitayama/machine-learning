-- num active at end
with cte as (
  select
    count(distinct account_id) as num_active
  from
    account_status
  where
    date = '2019-12-31'
    and status = 'open'
),
-- num close at start
cte2 as (
  select
    count(distinct account_id) as num_closed
  from
    account_status
  where
    date = '2020-01-01'
    and status = 'closed'

),
-- correct num closed by self-join
cte3 as (
  select
    count(*) as num_closed
  from
    account_status as a
  inner join
    account_status as b
  on
    a.account_id = b.account_id
  where
    a.date = '2019-12-31'
    and a.status = 'open'
    and b.date = '2020-01-01'
    and b.status = 'closed'
)

select
  round(
    (select num_closed from cte3)::decimal
    /
    (select num_active from cte),
    2
   ) as percentage_closed
;