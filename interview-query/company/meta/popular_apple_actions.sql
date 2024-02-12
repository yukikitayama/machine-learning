/*
count number of rows by action in 2020 nov data
assign ranks by dense_rank
output top 5 rows
*/

with
cte as (
  select
    action,
    count(*) as num_action
  from
    events
  where
    extract(month from created_at) = 11
    and extract(year from created_at) = 2020
    and platform in ('iphone', 'ipad')
  group by
    action
),

cte2 as (
  select
    *,
    dense_rank() over(
      order by num_action desc
    ) as ranks
  from
    cte
)

-- select * from events
-- select * from cte2;

select
  action,
  ranks
from
  cte2
order by
  ranks
limit
  5
;