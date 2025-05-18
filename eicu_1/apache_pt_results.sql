CREATE TEMP TABLE apache_pt_results AS
WITH ranked_apache AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY patientunitstayid ORDER BY apachescore DESC) AS rn,
         FIRST_VALUE(COALESCE(apacheversion, 'Unknown')) OVER (PARTITION BY patientunitstayid ORDER BY apachescore DESC) AS Apache_Version
  FROM `physionet-data.eicu_crd.apachepatientresult`
  WHERE apachescore IS NOT NULL AND apacheversion IS NOT NULL
),
pivoted AS (
    SELECT
        patientunitstayid,
        MAX(actualiculos) AS icu_los,
        MAX(actualhospitallos) AS hosp_los,
        MAX(actualventdays) AS daysonvent,
        MAX(apachescore) AS ApacheScore,
        MAX(predictedhospitalmortality) AS predicted_hospital_mortality,
        MAX(CASE
            WHEN REGEXP_CONTAINS(actualhospitalmortality, r'(?i)(EXPIRED)') THEN 1 ELSE 0
        END) AS hosp_mortality,
        MAX(CASE
            WHEN REGEXP_CONTAINS(actualicumortality, r'(?i)(EXPIRED)') THEN 1 ELSE 0
        END) AS icu_mortality,
        MAX(Apache_Version) AS Apache_Version
    FROM ranked_apache
    GROUP BY patientunitstayid
)
SELECT
    apache_vars.*,
    icu_los,
    hosp_los,
    daysonvent,
    hosp_mortality,
    icu_mortality,
    ApacheScore,
    Apache_Version,
    predicted_hospital_mortality,
    CASE
        WHEN hosp_mortality = 1 THEN 1
        WHEN icu_mortality = 1 THEN 1
        ELSE 0
    END AS mortality
FROM apache_vars
LEFT JOIN pivoted
    ON apache_vars.patientunitstayid = pivoted.patientunitstayid
-- WHERE hosp_mortality IS NOT NULL
    ;
