with cte as (
  select
    user_id as single_post_user_id
  from
    job_postings
  group by
    user_id
  having
    count(*) = count(distinct job_id)
)

select
  count(single_post_user_id) as single_post,
  (select count(distinct user_id) from job_postings) - count(single_post_user_id) as multiple_posts
from
  cte
;