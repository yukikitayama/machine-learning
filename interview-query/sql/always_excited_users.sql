with
-- Currently excited
cte1 as (
  select
    user_id,
    impression_id,
    row_number() over(
      partition by user_id
      order by dt desc
    ) as row_num
  from
    ad_impressions
),
cte2 as (
  select
    user_id
  from
    cte1
  where
    row_num = 1
    and impression_id = 'Excited'
),

-- Never bored
cte3 as (
  select
    user_id
  from
    ad_impressions
  group by
    1
  having
    max(if(impression_id = 'Bored', 1, 0)) = 0
)

select
  user_id
from
  cte2
where
  user_id in (select * from cte3)
;