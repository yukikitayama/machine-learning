with cte as (
  select
    concat(a.first_name, ' ', a.last_name) as employee_name,
    b.name as department_name,
    a.salary,
    dense_rank() over(
      partition by a.department_id
      order by a.salary desc
    ) as rnk
  from
    employees as a
  left join
    departments as b
  on
    a.department_id = b.id
)

select
  employee_name,
  department_name,
  salary
from
  cte
where
  rnk <= 3
order by
  2,
  3 desc
;