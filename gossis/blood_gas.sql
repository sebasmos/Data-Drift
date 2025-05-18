CREATE temp TABLE blood_gas as
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
        'paO2'
      , 'paCO2'
      , 'pH'
      , 'FiO2'
      )
      group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
      having count(distinct labresult)<=1
    )

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

      WHERE
         (lab.labname = 'paO2' and SAFE_CAST(labresult AS FLOAT64) >= 15 and SAFE_CAST(labresult AS FLOAT64) <= 720)
      OR (lab.labname = 'paCO2' and SAFE_CAST(labresult AS FLOAT64) >= 5 and SAFE_CAST(labresult AS FLOAT64) <= 250)
      OR (lab.labname = 'pH' and SAFE_CAST(labresult AS FLOAT64) >= 6.5 and SAFE_CAST(labresult AS FLOAT64) <= 8.5)
      OR (lab.labname = 'FiO2' and SAFE_CAST(labresult AS FLOAT64) >= 0.2 and SAFE_CAST(labresult AS FLOAT64) <= 1.0)
      -- we will fix fio2 units later
      OR (lab.labname = 'FiO2' and SAFE_CAST(labresult AS FLOAT64) >= 20 and SAFE_CAST(labresult AS FLOAT64) <= 100)
    )


select
    patientunitstayid
  , labresultoffset
  -- the aggregate (max()) only ever applies to 1 value due to the where clause
  , MAX(case
        when labname != 'FiO2' then null
        when SAFE_CAST(labresult AS FLOAT64) >= 20 then SAFE_CAST(labresult AS FLOAT64)/100.0
      else SAFE_CAST(labresult AS FLOAT64) end) as fio2
  , MAX (case when labname = 'paO2' then SAFE_CAST(labresult AS FLOAT64) else null end) as pao2
  , MAX(case when labname = 'paCO2' then SAFE_CAST(labresult AS FLOAT64) else null end) as paco2
  , MAX(case when labname = 'pH' then SAFE_CAST(labresult AS FLOAT64) else null end) as pH
from vw1
where rn = 1
group by patientunitstayid, labresultoffset
order by patientunitstayid, labresultoffset;
