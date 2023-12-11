select
  count(distinct user_id) as num_users_open_email
from
  events
where
  action = 'email_opened'
;
