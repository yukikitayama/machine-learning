-- Compute team size for each manager
with cte as (
  select
    manager_id,
    count(*) as team_size
  from
    employees
  group by
    1
)

select
  a.name as manager,
  b.team_size
from
  managers as a
left join
  cte as b
on
  a.id = b.manager_id
where
  -- Use subquery to find the largest team size
  -- and filter manager data
  b.team_size = (
    select
      max(team_size)
    from
      cte
  )
;