CREATE temp TABLE icustays AS
WITH pt AS
(
    SELECT
          pt.*
        , CASE
            WHEN pt.age = '' THEN NULL
            WHEN REGEXP_CONTAINS(age, '>') then cast(REPLACE(age, '>','') as int64) + 1
            else cast(age as INT64)
          END as age_num
        , CASE
            WHEN REGEXP_CONTAINS(pt.gender, r'(?i)(male)') THEN 1 ELSE 0
          END AS male
        , CASE
            WHEN REGEXP_CONTAINS(pt.gender, r'(?i)(female)') THEN 1 ELSE 0
          END AS female
        , ROW_NUMBER() over
         (
        PARTITION BY uniquepid
        ORDER BY
              hospitaldischargeyear
            , age
         ) as HOSP_NUM
    from `physionet-data.eicu_crd.patient` pt
)

, hospital as (
    SELECT
         hospitalid
        ,MAX(teachingstatus) as teachingstatus
        ,STRING_AGG(region) as region
        ,STRING_AGG(numbedscategory) as n_bed

    FROM `physionet-data.eicu_crd.hospital` AS hospital
    GROUP BY hospital.hospitalid
)
select
      pt.*
     ,hospital.teachingstatus
     ,hospital.region
     ,hospital.n_bed

FROM pt
LEFT JOIN hospital hospital
    ON pt.hospitalid = hospital.hospitalid

WHERE HOSP_NUM = 1
;