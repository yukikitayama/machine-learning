select
  student_name,
  sum(case when exam_id = 1 then score else null end) as exam_1,
  sum(case when exam_id = 2 then score else null end) as exam_2,
  sum(case when exam_id = 3 then score else null end) as exam_3,
  sum(case when exam_id = 4 then score else null end) as exam_4
from
  exam_scores
group by
  1
;