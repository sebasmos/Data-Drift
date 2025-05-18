DROP TABLE IF EXISTS gossis_bg_d1;
CREATE TABLE blood_gas_d1 AS
SELECT
    bg.patientunitstayid,
    MIN(PaO2 / NULLIF(fio2, 0)) AS PaO2FiO2Ratio_min,
    MAX(PaO2 / NULLIF(fio2, 0)) AS PaO2FiO2Ratio_max,
    MIN(PaO2) AS PaO2_min,
    MAX(PaO2) AS PaO2_max,
    MIN(PaCO2) AS PaCO2_min,
    MAX(PaCO2) AS PaCO2_max,
    MIN(pH) AS pH_min,
    MAX(pH) AS pH_max
FROM blood_gas AS bg
WHERE labresultoffset >= (-1*60) AND labresultoffset <= (24*60)
GROUP BY bg.patientunitstayid;
