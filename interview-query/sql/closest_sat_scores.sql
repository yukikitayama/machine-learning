with cte as (
  select
    a.student as one_student,
    b.student as other_student,
    abs(a.score - b.score) as score_diff,
    rank() over(
      order by abs(a.score - b.score)
    ) as rnk
  from
    scores as a,
    scores as b
  where
    a.id != b.id
    and a.student < b.student
)

select
  one_student,
  other_student,
  score_diff
from
  cte
where
  rnk = 1
limit
  1
;