CREATE temp TABLE apache_vars AS
WITH pivoted AS(
    SELECT
        patientunitstayid,
        CAST(MAX(intubated) AS INT64) AS intubated,
        CAST(MAX(vent) AS INT64) AS vent,
        CAST(MAX(dialysis) AS INT64) AS dialysis,
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
    FROM `lcp-consortium.eicu_crd_ii_v0_1_0.apacheapsvar`
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
