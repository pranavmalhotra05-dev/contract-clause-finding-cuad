-- Overlap between contracts: same company or same agreement type
select count(*) as contracts, count(distinct company) as companies, count(distinct agreement_type) as agreement_types from contracts;
select agreement_type, count(*) n from contracts group by 1 order by n desc limit 12;
select company, count(*) n from contracts group by 1 having n>1 order by n desc limit 10;
-- Clause prevalence per contract
select category, count(distinct contract_id) contracts_with, round(100.0*count(distinct contract_id)/510,1) pct,
       count(*) spans, round(avg(end-start)) mean_chars
from spans group by 1 order by pct desc;
