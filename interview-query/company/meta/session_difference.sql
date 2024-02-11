with cte as (
  select
    user_id,
    min(created_at) as first_session,
    max(created_at) as last_session
  from
    user_sessions
  where
    extract(year from created_at) = 2020
  group by
    user_id
)

-- select * from cte;

select
  user_id,
  date_part('day', last_session - first_session) as no_of_days
from
  cte
;