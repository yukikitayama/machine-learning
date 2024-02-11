/*
number of accepts divided by number of requests
*/

select
  round(
    count(b.acceptor_id)::decimal / count(a.requester_id),
    4
  ) as acceptance_rate
from
  friend_requests as a
left join
  friend_accepts as b
on
  a.requester_id = b.requester_id
  and a.requested_id = b.acceptor_id
;