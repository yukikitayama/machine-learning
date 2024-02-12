/*
We want cat to have a higher metric
  rating * 1 / position
*/


-- SELECT * FROM search_results

select
  query,
  round(avg(rating::decimal / position), 2) as avg_rating
from
  search_results
group by
  query
;

