select
  c.name,
  round(sum(b.price * a.quantity), 2) as total_cost,
  c.id as user_id
from
  transactions as a
left join
  products as b
on
  a.product_id = b.id
left join
  users as c
on
  a.user_id = c.id
group by
  1,
  3
order by
  2 desc
;
