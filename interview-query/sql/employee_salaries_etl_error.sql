with cte as (
  select
    first_name,
    last_name,
    salary,
    row_number() over(
      partition by first_name, last_name
      order by id desc
    ) row_num
  from
    employees
)

select
  first_name,
  last_name,
  salary
from
  cte
where
  row_num = 1
;