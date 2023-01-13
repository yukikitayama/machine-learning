select
  a.title as project_title_end,
  b.title as project_title_start,
  a.end_date as date
from
  projects as a
left join
  projects as b
on
  a.end_date = b.start_date
where
  b.title is not null
;
