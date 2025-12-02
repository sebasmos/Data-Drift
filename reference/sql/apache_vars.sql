CREATE temp TABLE apache_vars AS
WITH pivoted AS(
    SELECT
        patientunitstayid,
        CASE
            WHEN MAX(intubated) = 1 THEN 1
            ELSE 0
        END AS intubated,
        CASE
            WHEN MAX(vent) = 1 THEN 1
            ELSE 0
        END AS vent,
        CASE
            WHEN MAX(dialysis) = 1 THEN 1
            ELSE 0
        END AS dialysis,
        max(eyes) as eyes,
        max(motor) as motor,
        max(verbal) as verbal,
        max(meds) as meds,
        max(urine) as urine,
        max(wbc) as wbc,
        max(temperature) as temperature,
        max(respiratoryrate) as respiratoryrate,
        max(sodium) as sodium,
        max(heartrate) as heartrate,
        max(meanbp) as meanbp,
        max(ph) as ph,
        max(hematocrit) as hematocrit,
        max(creatinine) as creatinine,
        max(albumin) as albumin,
        max(pao2) as pao2,
        max(pco2) as pco2,
        max(bun) as bun,
        max(glucose) as glucose,
        max(bilirubin) as bilirubin,
        max(fio2) as fio2,
    FROM `physionet-data.eicu_crd.apacheapsvar`
    GROUP BY patientunitstayid
)

SELECT
        icustays.*,
        intubated,
        vent,
        dialysis,
        eyes,
        motor,
        verbal,
        meds,
        urine,
        wbc,
        temperature,
        respiratoryrate,
        sodium,
        heartrate,
        meanbp,
        ph,
        hematocrit,
        creatinine,
        albumin,
        pao2,
        pco2,
        bun,
        glucose,
        bilirubin,
        fio2

FROM icustays
LEFT JOIN pivoted
    ON icustays.patientunitstayid = pivoted.patientunitstayid
;
