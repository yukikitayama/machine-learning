with cte as (
  select
    query,
    count(rating) as num_ratings,
    sum(case when rating < 3 then 1 else 0 end) as num_bad_ratings
  from
    search_results
  group by
    1
)

-- select * from cte;

select
  round(
    avg(case when num_ratings = num_bad_ratings then 1 else 0 end),
    2
   ) as percentage_less_than_3
from
  cte
;