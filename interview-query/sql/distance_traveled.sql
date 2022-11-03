select
  a.name,
  -- If a user has no ride data, joining make null in distance column,
  -- but the problem requires us to report 0 if there was no travel
  ifnull(sum(b.distance)) as distance_traveled
from
  users as a
left join
  rides as b
on
  -- Inflates table by joining multiple rides in b to single user_id in a
  a.id = b.passenger_user_id
group by
  1
order by
  2 desc
;