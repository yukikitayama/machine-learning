select
  name
from
  neighborhoods
where
  id not in (
    select
      distinct neighborhood_id
    from
      users
  )
;