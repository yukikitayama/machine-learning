with cte as (
  select
    rank() over(
      partition by date
      order by downloads desc
    ) as daily_rank,
    user_id,
    date,
    downloads
  from
    download_facts
)

select
  *
from
  cte
where
  daily_rank <= 3
order by
  3,
  1
;