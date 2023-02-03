select
  a.shipment_id,
  a.ship_date,
  a.customer_id,
  if(
    a.ship_date between b.membership_start_date and b.membership_end_date,
    "Y",
    "N"
  ) as is_member,
  a.quantity
from
  shipments as a
left join
  customers as b
on
  a.customer_id = b.customer_id
;
