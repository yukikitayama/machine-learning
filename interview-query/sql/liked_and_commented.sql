select
  round(
    -- Number of users who never liked or commented
    count(id)
    /
    -- Total number of users
    (
      select
        count(id)
      from
        users
    ),
    2
  ) as percent_never
from
  users
where
  id not in (
    select
      user_id
    from
      events
    where
      action in ("like", "comment")
  )
;
