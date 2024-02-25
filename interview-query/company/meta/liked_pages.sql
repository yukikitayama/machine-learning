select
  a.user_id,
  b.page_id,
  count(distinct a.friend_id) as num_friend_likes
from
  friends as a
left join
  page_likes as b
on
  a.friend_id = b.user_id
where
  (a.user_id, b.page_id) not in (
    select user_id, page_id from page_likes
  )
group by
  1,
  2
;