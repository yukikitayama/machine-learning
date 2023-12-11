select
  timestampdiff(minute, start_dt, end_dt) as duration_minutes
from
  rides
where
  timestampdiff(minute, start_dt, end_dt) > 120
order by
  1 desc
;
