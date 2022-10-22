select
  b.sex,
  round(avg(a.quantity * c.price), 2) aov
from
  transactions as a
left join
  users as b
on
  a.user_id = b.id
left join
  products as c
on
  a.product_id = c.id
group by
  1
;