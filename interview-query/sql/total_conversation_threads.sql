with
cte1 as (
  select
    id,
    receiver_id,
    sender_id
  from
    messenger_sends
  union all
  select
    id,
    sender_id as receiver_id,
    receiver_id as sender_id
  from
    messenger_sends
),
cte2 as (
select
  distinct receiver_id,
  sender_id
from
  cte1
where
  receiver_id < sender_id
)

select
  count(*) as total_conv_threads
from
  cte2
;
