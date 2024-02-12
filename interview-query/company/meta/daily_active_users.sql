/*
what is daily active users
  users who had logs on a particular date

By platform and created_at, count the distinct user_id in 2020
*/

select
  platform,
  created_at,
  count(distinct user_id) as daily_users
from
  events
where
  extract(year from created_at) = 2020
group by
  platform,
  created_at
;