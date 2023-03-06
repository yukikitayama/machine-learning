-- Two sample t test for comparing two means
-- Compute mean, var, number of data of category 9
-- Compute mean, var, number of data of category except 9
-- Compute degree of freedom by n1 + n2 - 2

with
cte1 as (
  select
    avg(price) as mean1,
    var_samp(price) as var1,
    count(price) as num1
  from
    products
  where
    category_id = 9
),
cte2 as (
  select
    avg(price) as mean2,
    var_samp(price) as var2,
    count(price) as num2
  from
    products
  where
    category_id != 9
),
cte3 as (
  select
    *
  from
    cte1,
    cte2
)

select
  (mean1 - mean2) / sqrt((var1 / num1) + (var2 / num2)) as t_value,
  num1 + num2 - 2 as d_o_f
from
  cte3
;