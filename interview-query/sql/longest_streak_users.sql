with
-- Remove duplicated dates
cte1 as (
  select
    user_id,
    created_at
  from
    events
  group by
    1,
    2
),

-- Create group ID
cte2 as (
  select
    user_id,
    date_sub(
      created_at,
      interval
        row_number() over(
          partition by user_id
          order by created_at
        ) day
    ) as group_id
  from
    cte1
),

-- Count longest continuous streak
cte3 as (
  select
    user_id,
    group_id,
    count(*) as cnt
  from
    cte2
  group by
    1,
    2
),
cte4 as (
  select
    user_id,
    max(cnt) as longest_continuous_streak
  from
    cte3
  group by
    1
)

select
  user_id,
  longest_continuous_streak as streak_length
from
  cte4
order by
  2 desc
limit 5
;
