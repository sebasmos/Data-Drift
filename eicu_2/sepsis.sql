create temp table sepsis as
WITH dx_sepsis AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN REGEXP_CONTAINS(diagnosisstring, r'(?i)(sepsis)|(septic)') THEN 1
            ELSE 0
          END) AS sepsis
    FROM `lcp-consortium.eicu_crd_ii_v0_1_0.diagnosis`
    WHERE REGEXP_CONTAINS(diagnosisstring, '(?i)((^|[|])onc)|' ||
                                            'cardiovascular|shock / hypotension|sepsis' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with multi-organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- acute renal failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- acute respiratory failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- circulatory system failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- congestive heart failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- critical care myopathy' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- critical care neuropathy' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- metabolic encephalopathy' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction-acute hepatic failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|severe' ||
                                            'cardiovascular|shock / hypotension|sepsis|severe|with septic shock' ||
                                            'cardiovascular|shock / hypotension|sepsis|severe|without septic shock' ||
                                            'cardiovascular|shock / hypotension|septic shock' ||
                                            'cardiovascular|shock / hypotension|septic shock|culture negative' ||
                                            'cardiovascular|shock / hypotension|septic shock|cultures pending' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|fungal' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|gram negative organism' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|gram positive organism' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|parasitic' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process with organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process with organ dysfunction|non-infectious origin with acute organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction|systemic inflammatory response syndrome (SIRS) of non-infectious origin witho' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction|unspecified organism' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to non-infectious process with organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to non-infectious process without organ dysfunction' ||
                                            'cardiovascular|vascular disorders|arterial thromboembolism|due to sepsis' ||
                                            'cardiovascular|vascular disorders|peripheral vascular ischemia|due to sepsis' ||
                                            'endocrine|fluids and electrolytes|hypocalcemia|due to sepsis' ||
                                            'hematology|coagulation disorders|DIC syndrome|associated with sepsis/septic shock' ||
                                            'infectious diseases|systemic/other infections|sepsis' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with multi-organ dysfunction syndrome' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- acute renal failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- acute respiratory failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- circulatory system failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- congestive heart failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- critical care myopathy' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- critical care neuropathy' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- metabolic encephalopathy' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction-acute hepatic failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|severe' ||
                                            'infectious diseases|systemic/other infections|sepsis|severe|septic shock' ||
                                            'infectious diseases|systemic/other infections|sepsis|severe|without septic shock' ||
                                            'infectious diseases|systemic/other infections|septic shock' ||
                                            'infectious diseases|systemic/other infections|septic shock|culture negative' ||
                                            'infectious diseases|systemic/other infections|septic shock|cultures pending' ||
                                            'infectious diseases|systemic/other infections|septic shock|fungal' ||
                                            'infectious diseases|systemic/other infections|septic shock|gram negative organism' ||
                                            'infectious diseases|systemic/other infections|septic shock|gram positive organism' ||
                                            'infectious diseases|systemic/other infections|septic shock|organism identified' ||
                                            'infectious diseases|systemic/other infections|septic shock|parasitic' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to infectious process with organ dysfunction' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to non-infectious process with organ dysfunction' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to non-infectious process without organ dysfunction' ||
                                            'pulmonary|respiratory failure|acute lung injury|non-pulmonary etiology|sepsis' ||
                                            'pulmonary|respiratory failure|ARDS|non-pulmonary etiology|sepsis' ||
                                            'renal|disorder of kidney|acute renal failure|due to sepsis' ||
                                            'renal|electrolyte imbalance|hypocalcemia|due to sepsis')
    GROUP BY patientunitstayid
)
, admit_sepsis AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN REGEXP_CONTAINS(admitdxpath, r'(?i)(sepsis)|(septic)') THEN 1
            ELSE 0
          END) AS sepsis
    FROM `lcp-consortium.eicu_crd_ii_v0_1_0.admissiondx`
    WHERE REGEXP_CONTAINS(admitdxpath, r'(?i)((^|[|])onc)|' ||
                                            'cardiovascular|shock / hypotension|sepsis' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with multi-organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- acute renal failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- acute respiratory failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- circulatory system failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- congestive heart failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- critical care myopathy' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- critical care neuropathy' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction- metabolic encephalopathy' ||
                                            'cardiovascular|shock / hypotension|sepsis|sepsis with single organ dysfunction-acute hepatic failure' ||
                                            'cardiovascular|shock / hypotension|sepsis|severe' ||
                                            'cardiovascular|shock / hypotension|sepsis|severe|with septic shock' ||
                                            'cardiovascular|shock / hypotension|sepsis|severe|without septic shock' ||
                                            'cardiovascular|shock / hypotension|septic shock' ||
                                            'cardiovascular|shock / hypotension|septic shock|culture negative' ||
                                            'cardiovascular|shock / hypotension|septic shock|cultures pending' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|fungal' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|gram negative organism' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|gram positive organism' ||
                                            'cardiovascular|shock / hypotension|septic shock|organism identified|parasitic' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process with organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process with organ dysfunction|non-infectious origin with acute organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction|systemic inflammatory response syndrome (SIRS) of non-infectious origin witho' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction|unspecified organism' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to non-infectious process with organ dysfunction' ||
                                            'cardiovascular|shock / hypotension|signs and symptoms of sepsis (SIRS)|due to non-infectious process without organ dysfunction' ||
                                            'cardiovascular|vascular disorders|arterial thromboembolism|due to sepsis' ||
                                            'cardiovascular|vascular disorders|peripheral vascular ischemia|due to sepsis' ||
                                            'endocrine|fluids and electrolytes|hypocalcemia|due to sepsis' ||
                                            'hematology|coagulation disorders|DIC syndrome|associated with sepsis/septic shock' ||
                                            'infectious diseases|systemic/other infections|sepsis' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with multi-organ dysfunction syndrome' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- acute renal failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- acute respiratory failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- circulatory system failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- congestive heart failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- critical care myopathy' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- critical care neuropathy' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction- metabolic encephalopathy' ||
                                            'infectious diseases|systemic/other infections|sepsis|sepsis with single organ dysfunction-acute hepatic failure' ||
                                            'infectious diseases|systemic/other infections|sepsis|severe' ||
                                            'infectious diseases|systemic/other infections|sepsis|severe|septic shock' ||
                                            'infectious diseases|systemic/other infections|sepsis|severe|without septic shock' ||
                                            'infectious diseases|systemic/other infections|septic shock' ||
                                            'infectious diseases|systemic/other infections|septic shock|culture negative' ||
                                            'infectious diseases|systemic/other infections|septic shock|cultures pending' ||
                                            'infectious diseases|systemic/other infections|septic shock|fungal' ||
                                            'infectious diseases|systemic/other infections|septic shock|gram negative organism' ||
                                            'infectious diseases|systemic/other infections|septic shock|gram positive organism' ||
                                            'infectious diseases|systemic/other infections|septic shock|organism identified' ||
                                            'infectious diseases|systemic/other infections|septic shock|parasitic' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to infectious process with organ dysfunction' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to infectious process without organ dysfunction' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to non-infectious process with organ dysfunction' ||
                                            'infectious diseases|systemic/other infections|signs and symptoms of sepsis (SIRS)|due to non-infectious process without organ dysfunction' ||
                                            'pulmonary|respiratory failure|acute lung injury|non-pulmonary etiology|sepsis' ||
                                            'pulmonary|respiratory failure|ARDS|non-pulmonary etiology|sepsis' ||
                                            'renal|disorder of kidney|acute renal failure|due to sepsis' ||
                                            'renal|electrolyte imbalance|hypocalcemia|due to sepsis')
    GROUP BY patientunitstayid
)
, icd_sepsis AS
(
    SELECT
          patientunitstayid
        , MAX(CASE
            WHEN SUBSTR(icd9code,0,3) BETWEEN '785.51' AND '995.92' THEN 1
            ELSE 0
          END) AS sepsis
    FROM `lcp-consortium.eicu_crd_ii_v0_1_0.diagnosis`
    WHERE SUBSTR(icd9code,0,3) LIKE '995.91'
    OR SUBSTR(icd9code,0,3) LIKE '785.52'
    GROUP BY patientunitstayid
)

SELECT
      apr.*
      ,CASE
        WHEN dx_sepsis.sepsis = 1 THEN 1
        WHEN icd_sepsis.sepsis = 1 THEN 1
        WHEN admit_sepsis.sepsis = 1 THEN 1
        ELSE 0
      END AS has_sepsis


FROM apache_pt_results apr
LEFT JOIN dx_sepsis dx_sepsis
    ON apr.patientunitstayid = dx_sepsis.patientunitstayid
LEFT JOIN icd_sepsis icd_sepsis
    ON apr.patientunitstayid = icd_sepsis.patientunitstayid
LEFT JOIN admit_sepsis admit_sepsis
    ON apr.patientunitstayid = admit_sepsis.patientunitstayid

;




