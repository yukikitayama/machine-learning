select
  distinct make
from
  cars
order by
  rand()
limit
  1
;
