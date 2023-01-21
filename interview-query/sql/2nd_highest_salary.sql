with cte as (
  select
    salary,
    dense_rank() over(order by salary desc) as rnk
  from
    employees as a
  left join
    departments as b
  on
    a.department_id = b.id
  where
    b.name = 'engineering'
)

select
  salary
from
  cte
where
  rnk = 2
;

