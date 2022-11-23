select
  passenger_user_id,
  -- TIMESTAMPDIFF(unit, start_datetime, end_datetime)
  avg(timestampdiff(minute, start_dt, end_dt)) as avg_time
from
  rides
group by
  1
;