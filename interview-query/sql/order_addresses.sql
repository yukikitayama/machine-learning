select
  avg(if(a.shipping_address = b.address, 1, 0)) as home_address_percent
from
  transactions as a
left join
  users as b
on
  a.user_id = b.id
;