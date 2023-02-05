with
cte1 as (
  select
    user_id,
    created_at
  from
    comments
  where
    year(created_at) = 2020
    and month(created_at) = 1
),
cte2 as (
  select
    a.id,
    ifnull(count(b.created_at), 0) as num_comment
  from
    users as a
  left join
    cte1 as b
  on
    a.id = b.user_id
  group by
    1
)

select
  num_comment as comment_count,
  count(*) as frequency
from
  cte2
group by
  1
;
