-- name: vomiting_per_week
-- Vomiting episodes per patient per treatment week (week 1 = first 7 days of treatment).
SELECT d.patient_id,
       p.display_name,
       CAST((julianday(d.day_date) - julianday(p.treatment_start)) / 7 AS INTEGER) + 1 AS treatment_week,
       SUM(d.display_value)                          AS vomiting_episodes,
       SUM(CASE WHEN d.display_value > 0 THEN 1 ELSE 0 END) AS days_with_vomiting
FROM daily_symptoms d JOIN patients p USING (patient_id)
WHERE d.symptom = 'vomiting'
GROUP BY d.patient_id, treatment_week
ORDER BY d.patient_id, treatment_week;

-- name: nausea_around_med_change
-- Average questionnaire nausea score (0-4) in the 7 days before vs the 7 days after a
-- prescriber started a new antiemetic mid-course. Descriptive only: no causal claim.
WITH changes AS (
    SELECT me.patient_id, me.med_code, me.event_date
    FROM medication_events me
    JOIN medications m USING (med_code)
    JOIN patients p USING (patient_id)
    WHERE m.med_class = 'antiemetic' AND me.event_type = 'start'
      AND me.event_date > p.treatment_start
)
SELECT c.patient_id, c.med_code AS started, c.event_date AS start_date,
       ROUND(AVG(CASE WHEN d.day_date <  c.event_date THEN d.display_value END), 2) AS avg_nausea_7d_before,
       ROUND(AVG(CASE WHEN d.day_date >= c.event_date THEN d.display_value END), 2) AS avg_nausea_7d_after,
       SUM(CASE WHEN d.day_date <  c.event_date THEN 1 ELSE 0 END) AS days_before,
       SUM(CASE WHEN d.day_date >= c.event_date THEN 1 ELSE 0 END) AS days_after
FROM changes c
JOIN daily_symptoms d
  ON d.patient_id = c.patient_id AND d.symptom = 'nausea'
 AND d.value_source IN ('questionnaire', 'both')
 AND d.day_date BETWEEN date(c.event_date, '-7 day') AND date(c.event_date, '+6 day')
GROUP BY c.patient_id, c.med_code, c.event_date;

-- name: nausea_by_days_since_cisplatin
-- Average nausea score by days since the most recent cisplatin dose, all patients.
-- Shows the post-chemo pattern a provider would expect to see (or not).
WITH tagged AS (
    SELECT d.patient_id, d.day_date, d.display_value,
           (SELECT MAX(t.event_date) FROM treatment_events t
             WHERE t.patient_id = d.patient_id AND t.type = 'cisplatin' AND t.event_date <= d.day_date) AS last_dose
    FROM daily_symptoms d
    WHERE d.symptom = 'nausea' AND d.value_source IN ('questionnaire', 'both')
)
SELECT CAST(julianday(day_date) - julianday(last_dose) AS INTEGER) AS days_since_cisplatin,
       ROUND(AVG(display_value), 2) AS avg_nausea_0_4,
       COUNT(*) AS patient_days
FROM tagged
WHERE last_dose IS NOT NULL
GROUP BY days_since_cisplatin
ORDER BY days_since_cisplatin;

-- name: weight_vs_alert_line
-- Latest weight, change since day 1, and the first day each patient crossed the clinic's alert line.
WITH w AS (
    SELECT m.patient_id, date(m.measured_at) AS day, m.value,
           ROW_NUMBER() OVER (PARTITION BY m.patient_id ORDER BY m.measured_at DESC) AS rn
    FROM measurements m WHERE m.type = 'weight_lb'
)
SELECT p.patient_id, p.display_name, p.baseline_weight_lb, p.alert_line_lb,
       (SELECT value FROM w WHERE w.patient_id = p.patient_id AND rn = 1) AS latest_weight_lb,
       ROUND((SELECT value FROM w WHERE w.patient_id = p.patient_id AND rn = 1) - p.baseline_weight_lb, 1) AS change_lb,
       ROUND(100.0 * ((SELECT value FROM w WHERE w.patient_id = p.patient_id AND rn = 1) - p.baseline_weight_lb)
             / p.baseline_weight_lb, 1) AS change_pct,
       (SELECT MIN(day) FROM w WHERE w.patient_id = p.patient_id AND w.value < p.alert_line_lb) AS first_below_alert
FROM patients p
ORDER BY change_pct;

-- name: checkin_adherence
-- How consistently each patient reports. Missing days are shown as missing, never as "no symptoms".
WITH days AS (
    SELECT patient_id, date(submitted_at) AS day FROM reports GROUP BY patient_id, day
),
span AS (
    SELECT patient_id, treatment_start,
           (SELECT MAX(date(submitted_at)) FROM reports) AS as_of
    FROM patients
)
SELECT s.patient_id,
       CAST(julianday(s.as_of) - julianday(s.treatment_start) AS INTEGER) + 1 AS days_on_treatment,
       COUNT(d.day) AS days_with_checkin,
       ROUND(100.0 * COUNT(d.day) / (julianday(s.as_of) - julianday(s.treatment_start) + 1), 0) AS checkin_rate_pct,
       CAST(julianday(s.as_of) - julianday(MAX(d.day)) AS INTEGER) AS days_since_last_checkin
FROM span s LEFT JOIN days d USING (patient_id)
GROUP BY s.patient_id
ORDER BY checkin_rate_pct;

-- name: changed_since_last_visit
-- Worst score per symptom since the last visit vs the same-length window before it.
-- :patient_id and :last_visit are parameters.
WITH win AS (
    SELECT :last_visit AS lv,
           CAST(julianday((SELECT MAX(day_date) FROM daily_symptoms WHERE patient_id = :patient_id))
                - julianday(:last_visit) AS INTEGER) AS n
)
SELECT d.symptom,
       MAX(CASE WHEN d.day_date <  (SELECT lv FROM win)
                 AND d.day_date >= date((SELECT lv FROM win), '-' || (SELECT n FROM win) || ' day')
                THEN d.display_value END) AS worst_before,
       MAX(CASE WHEN d.day_date >= (SELECT lv FROM win) THEN d.display_value END) AS worst_since,
       SUM(CASE WHEN d.day_date >= (SELECT lv FROM win) THEN d.conflict ELSE 0 END) AS conflicts_since
FROM daily_symptoms d
WHERE d.patient_id = :patient_id AND d.display_value IS NOT NULL
GROUP BY d.symptom;

-- name: review_workload
-- What still needs a human: flagged extractions, rejected items and same-day conflicts per patient.
SELECT p.patient_id, p.display_name,
       (SELECT COUNT(*) FROM review_queue q WHERE q.patient_id = p.patient_id) AS awaiting_review,
       (SELECT COUNT(*) FROM rejected_extractions r JOIN reports USING (report_id)
         WHERE reports.patient_id = p.patient_id) AS rejected,
       (SELECT COUNT(*) FROM daily_symptoms d WHERE d.patient_id = p.patient_id AND d.conflict = 1) AS conflicting_days
FROM patients p;
