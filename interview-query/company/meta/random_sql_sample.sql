
with cte as (
  select
    ceil(random() * (select max(id) from big_table)) as random_id
)

-- select * from cte;

select
  a.id,
  a.name
from
  big_table as a
inner join
  cte as b
on
  -- If there are missing ID, we can't have exact match
  -- so use >= to guarantee to have some kind of ID
  a.id >= b.random_id
order by
  a.id
limit
  1
;
