select
  b.name as department_name,
  count(*) as number_of_employees,
  -- AVG() gives us the percentage by averaging true or false
  avg(a.salary > 100000) as percentage_over_100k
from
  employees as a
left join
  departments as b
on
  a.department_id = b.id
group by
  a.department_id
having
  -- Department with at least ten employees
  count(*) >= 10
order by
  -- Rank by the percentage of employees over 100k salary
  3
limit
  -- Find top 3
  3
;