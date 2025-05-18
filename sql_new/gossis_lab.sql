CREATE temp TABLE gossis_lab as
-- remove duplicate labs if they exist at the same time
with vw0 as
(
  select
      patientunitstayid
    , labname
    , labresultoffset
    , labresultrevisedoffset
  from `lcp-consortium.eicu_crd_ii_v0_1_0.lab` as lab
  where labname in
  (
    'albumin'
    , 'total bilirubin'
    , 'BUN'
    , 'calcium'
    , 'creatinine'
    , 'bedside glucose', 'glucose'
    , 'bicarbonate' -- HCO3
    -- TODO: what about 'Total CO2'
    , 'Hct'
    , 'Hgb'
    , 'PT - INR'
    , 'lactate'
    , 'platelets x 1000'
    , 'potassium'
    , 'sodium'
    , 'WBC x 1000'
  )
  group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
  having count(distinct labresult)<=1
)
-- get the last lab to be revised
, vw1 as
(
  select
      lab.patientunitstayid
    , lab.labname
    , lab.labresultoffset
    , lab.labresultrevisedoffset
    , lab.labresult
    , ROW_NUMBER() OVER
        (
          PARTITION BY lab.patientunitstayid, lab.labname, lab.labresultoffset
          ORDER BY lab.labresultrevisedoffset DESC
        ) as rn
  from `lcp-consortium.eicu_crd_ii_v0_1_0.lab` as lab
  inner join vw0
    ON  lab.patientunitstayid = vw0.patientunitstayid
    AND lab.labname = vw0.labname
    AND lab.labresultoffset = vw0.labresultoffset
    AND lab.labresultrevisedoffset = vw0.labresultrevisedoffset
  -- only valid lab values
  WHERE
       (lab.labname = 'albumin' and SAFE_CAST(lab.labresult AS FLOAT64) >= 0.5 and SAFE_CAST(lab.labresult AS FLOAT64) <= 6.5)
    OR (lab.labname = 'total bilirubin' and SAFE_CAST(lab.labresult AS FLOAT64) >= 0.2 and SAFE_CAST(lab.labresult AS FLOAT64)  <= 70.175)
    OR (lab.labname = 'BUN' and SAFE_CAST(lab.labresult AS FLOAT64) >= 1 and SAFE_CAST(lab.labresult AS FLOAT64) <= 280)
    OR (lab.labname = 'calcium' and SAFE_CAST(lab.labresult AS FLOAT64) > 0 and SAFE_CAST(lab.labresult AS FLOAT64) <= 9999)
    OR (lab.labname = 'creatinine' and SAFE_CAST(lab.labresult AS FLOAT64) >= 0.1 and SAFE_CAST(lab.labresult AS FLOAT64) <= 28.28)
    OR (lab.labname in ('bedside glucose', 'glucose') and SAFE_CAST(lab.labresult AS FLOAT64) >= 1.8018 and SAFE_CAST(lab.labresult AS FLOAT64) <= 1621.6)
    OR (lab.labname = 'bicarbonate' and SAFE_CAST(lab.labresult AS FLOAT64) >= 0 and SAFE_CAST(lab.labresult AS FLOAT64) <= 9999)
    -- will convert hct unit to fraction later
    OR (lab.labname = 'Hct' and SAFE_CAST(lab.labresult AS FLOAT64) >= 5 and SAFE_CAST(lab.labresult AS FLOAT64) <= 75)
    OR (lab.labname = 'Hgb' and SAFE_CAST(lab.labresult AS FLOAT64) > 0 and SAFE_CAST(lab.labresult AS FLOAT64) <= 9999)
    OR (lab.labname = 'PT - INR' and SAFE_CAST(lab.labresult AS FLOAT64) > 0 and SAFE_CAST(lab.labresult AS FLOAT64) <= 9999)
    OR (lab.labname = 'lactate' and SAFE_CAST(lab.labresult AS FLOAT64) > 0 and SAFE_CAST(lab.labresult AS FLOAT64) <= 9999)
    OR (lab.labname = 'platelets x 1000' and SAFE_CAST(lab.labresult AS FLOAT64) > 0 and SAFE_CAST(lab.labresult AS FLOAT64) <= 9999)
    OR (lab.labname = 'potassium' and SAFE_CAST(lab.labresult AS FLOAT64) >= 0.05 and SAFE_CAST(lab.labresult AS FLOAT64) <= 12)
    OR (lab.labname = 'sodium' and SAFE_CAST(lab.labresult AS FLOAT64) >= 90 and SAFE_CAST(lab.labresult AS FLOAT64) <= 215)
    OR (lab.labname = 'WBC x 1000' and SAFE_CAST(lab.labresult AS FLOAT64) >= 0 and SAFE_CAST(lab.labresult AS FLOAT64) <= 300)
)


select
    patientunitstayid
  , labresultoffset
  -- the aggregate (max()) only ever applies to 1 value due to the where clause
  , MAX(case when labname = 'albumin' then labresult else null end) as albumin
  , MAX(case when labname = 'total bilirubin' then labresult else null end) as bilirubin
  , MAX(case when labname = 'BUN' then labresult else null end) as BUN
  , MAX(case when labname = 'calcium' then labresult else null end) as calcium
  , MAX(case when labname = 'creatinine' then labresult else null end) as creatinine
  , MAX(case when labname in ('bedside glucose', 'glucose') then labresult else null end) as glucose
  , MAX(case when labname = 'bicarbonate' then labresult else null end) as hco3 -- bicarbonate
  , MAX(case when labname = 'Hct' then labresult else null end) as hematocrit
  , MAX(case when labname = 'Hgb' then labresult else null end) as hemoglobin
  , MAX(case when labname = 'PT - INR' then labresult else null end) as INR
  , MAX(case when labname = 'lactate' then labresult else null end) as lactate
  , MAX(case when labname = 'platelets x 1000' then labresult else null end) as platelets
  , MAX(case when labname = 'potassium' then labresult else null end) as potassium
  , MAX(case when labname = 'sodium' then labresult else null end) as sodium
  , MAX(case when labname = 'WBC x 1000' then labresult else null end) as wbc
from vw1
where rn = 1
group by patientunitstayid, labresultoffset
order by patientunitstayid, labresultoffset;
