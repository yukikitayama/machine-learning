with
cte1 as (
  select
    project_id,
    count(distinct employee_id) as employee_count
  from
    employee_projects
  group by
    1
),
cte2 as (
  select
    a.title,
    a.budget / b.employee_count as budget_per_employee,
    dense_rank() over(
      order by a.budget / b.employee_count desc
    ) as rnk
  from
    projects as a
  left join
    cte1 as b
  on
    a.id = b.project_id
  where
    b.employee_count is not null
)

select
  title,
  budget_per_employee
from
  cte2
where
  rnk <= 5
;
