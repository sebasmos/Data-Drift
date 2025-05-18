CREATE temp TABLE vitals AS
WITH pivoted AS(
    SELECT
        patientunitstayid,
        MAX(temperature) as max_temperature,
        MIN(temperature) as min_temperature,
        AVG(temperature) as mean_temperature,
        MAX(heartrate) as max_heartrate,
        MIN(heartrate) as min_heartrate,
        AVG(heartrate) as mean_heartrate,
        MAX(respiration) as max_respiration,
        MIN(respiration) as min_respiration,
        AVG(respiration) as mean_respiration
    FROM `physionet-data.eicu_crd.vitalperiodic`
    GROUP BY patientunitstayid
)

SELECT
    apache_vars.*,
    max_temperature,
    min_temperature,
    mean_temperature,
    max_heartrate,
    min_heartrate,
    mean_heartrate,
    max_respiration,
    min_respiration,
    mean_respiration

FROM apache_vars
LEFT JOIN pivoted
    ON apache_vars.patientunitstayid = pivoted.patientunitstayid
;
