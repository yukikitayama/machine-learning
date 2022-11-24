select
  a.download_date,
  b.paying_customer,
  round(avg(a.downloads), 2) as average_downloads
from
  downloads as a
left join
  accounts as b
on
  a.account_id = b.account_id
group by
  1,
  2
;
