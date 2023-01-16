select
  concat(a.first_name, ' ', a.last_name) as employee_name
from
  employees as a
left join
  managers as b
on
  a.manager_id = b.id
where
  a.join_date < b.join_date
;
