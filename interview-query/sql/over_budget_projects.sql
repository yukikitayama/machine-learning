with cte as (
  select
    a.title,
    a.budget,
    datediff(a.end_date, a.start_date) as project_days,
    sum(coalesce(c.salary, 0)) as total_salary
  from
    projects as a
  left join
    employee_projects as b
  on
    a.id = b.project_id
  left join
    employees as c
  on
    b.employee_id = c.id
  group by
    1,
    2,
    3
)

select
  title,
  -- Salary prorated to the day
  case
    when total_salary * (project_days / 365) > budget then "overbudget"
    else "within budget"
  end as project_forecast
from
  cte
;