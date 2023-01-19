with cte as (
  select
    a.title,
    a.budget / count(distinct b.employee_id) as budget_per_employee,
    dense_rank() over(order by a.budget / count(distinct b.employee_id) desc) as rnk
  from
    projects as a
  left join
    employee_projects as b
  on
    a.id = b.project_id
  group by
    1
)

select
  title,
  budget_per_employee
from
  cte
where
  rnk <= 5
;
