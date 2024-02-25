with cte1 as (
  select
    distinct user_id,
    url,
    date(created_at) as visit_date
  from
    events
  order by
    1,
    2,
    3
),

cte2 as (
  select
    *,
    row_number() over(
      partition by user_id, url
      order by visit_date
    ) as row_num
  from
    cte1
),

cte3 as (
  select
    *,
    -- Consecutive date assigned same date
    visit_date - interval '1 day' * row_num as group_date
  from
    cte2
),

cte4 as (
  select
    user_id,
    url,
    group_date,
    count(*) as day_streaks
  from
    cte3
  group by
    1,
    2,
    3
),

cte5 as (
  select
    count(distinct user_id) as num_frequent_user
  from
    cte4
  where
    day_streaks >= 7
)

-- select * from cte5;

select
  round(
    (select num_frequent_user from cte5)::decimal
    /
    count(distinct user_id),
    2
  ) as percent_of_users
from
  cte4
;

-- ----------------------------------------------------

WITH unique_visits AS (
    SELECT DISTINCT user_id
      , url
      , DATE(created_at) AS date_created
    FROM events
    ORDER BY user_id
      , url
      , date_created
)

, date_groups AS (
    SELECT user_id
        , url
        , date_created
        , RANK() OVER (PARTITION BY url,user_id ORDER BY date_created) AS row_num
    FROM unique_visits
)

, date_groups_2 AS (
    SELECT user_id
        , url
        , date_created
        , row_num
        , date_created - INTERVAL '1 DAY' * row_num AS date_group
    FROM date_groups
)

, user_streaks AS (
    SELECT user_id
        , url
        , date_group
        , COUNT(*) AS day_streaks
    FROM date_groups_2
    GROUP BY user_id
    , url
    , date_group
)

SELECT
    ROUND(
    (SELECT COUNT(DISTINCT user_id)
     FROM (SELECT user_id
           FROM user_streaks
           GROUP BY user_id
           HAVING MAX(day_streaks) >= 7) AS active_users
           ) * 1.0
     / COUNT(DISTINCT user_id)
     , 2) AS precent_of_users
FROM user_streaks