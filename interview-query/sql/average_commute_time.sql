with
cte1 as (
  select
    city,
    floor(avg(timestampdiff(minute, start_dt, end_dt))) as avg_time
  from
    rides
  where
    city = "NY"
  group by
    1
),
cte2 as (
  select
    city,
    commuter_id,
    floor(avg(timestampdiff(minute, start_dt, end_dt))) as avg_commuter_time
  from
    rides
  where
    city = "NY"
  group by
    1,
    2
)

select
  a.commuter_id,
  a.avg_commuter_time,
  b.avg_time
from
  cte2 as a
left join
  cte1 as b
on
  a.city = b.city
;