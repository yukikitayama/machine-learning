select
  date(created_at) as 'date',
  sum(count(id)) over(
    partition by year(created_at), month(created_at)
    order by created_at
  ) as monthly_cumulative
from
  users
group by
  1
order by
  1
;