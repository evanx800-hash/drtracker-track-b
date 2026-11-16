-- Dr.Tracker Track B prototype schema (SQLite).
-- Event-based: one row per thing the patient reported, never one wide row per day.
-- Raw reports are immutable; every structured row links back to the report it came from.

PRAGMA foreign_keys = ON;

CREATE TABLE patients (
    patient_id          TEXT PRIMARY KEY,
    display_name        TEXT NOT NULL,          -- synthetic
    age                 INTEGER,
    diagnosis           TEXT,
    regimen             TEXT,
    treatment_start     DATE NOT NULL,
    baseline_weight_lb  REAL,
    alert_line_lb       REAL,                   -- set by the clinic, as in the demo console
    last_visit_date     DATE,
    synthetic_note      TEXT
);

-- Clinic-side facts: radiation sessions and chemo doses.
CREATE TABLE treatment_events (
    event_id    INTEGER PRIMARY KEY,
    patient_id  TEXT NOT NULL REFERENCES patients(patient_id),
    event_date  DATE NOT NULL,
    type        TEXT NOT NULL CHECK (type IN ('radiation', 'cisplatin')),
    detail      TEXT
);

CREATE TABLE medications (
    med_code    TEXT PRIMARY KEY,
    med_class   TEXT NOT NULL
);

-- What the prescriber changed (start / stop / dose change).
CREATE TABLE medication_events (
    event_id    INTEGER PRIMARY KEY,
    patient_id  TEXT NOT NULL REFERENCES patients(patient_id),
    med_code    TEXT NOT NULL REFERENCES medications(med_code),
    event_date  DATE NOT NULL,
    event_type  TEXT NOT NULL CHECK (event_type IN ('start', 'stop', 'dose_change')),
    dose        TEXT,
    note        TEXT
);

-- Every submission exactly as received. Never edited.
CREATE TABLE reports (
    report_id     TEXT PRIMARY KEY,
    patient_id    TEXT NOT NULL REFERENCES patients(patient_id),
    submitted_at  TIMESTAMP NOT NULL,
    source        TEXT NOT NULL CHECK (source IN ('questionnaire', 'free_text')),
    raw_text      TEXT,          -- free text as typed / transcribed
    raw_answers   TEXT           -- questionnaire answers as JSON
);

-- One row per symptom statement. Absence ("no vomiting") is stored as present = 0,
-- which is different from the symptom simply not being mentioned (no row).
CREATE TABLE symptom_observations (
    obs_id            INTEGER PRIMARY KEY,
    report_id         TEXT NOT NULL REFERENCES reports(report_id),
    patient_id        TEXT NOT NULL REFERENCES patients(patient_id),
    symptom           TEXT NOT NULL,
    present           INTEGER NOT NULL CHECK (present IN (0, 1)),
    score             REAL,          -- questionnaire score or a number the patient typed; NULL otherwise
    intensity         TEXT CHECK (intensity IN ('none', 'mild', 'moderate', 'severe')),  -- patient's words
    episode_count     INTEGER,       -- vomiting episodes; NULL when the patient was vague ("a few")
    occurred_date     DATE NOT NULL, -- report date + relative day
    time_of_day       TEXT,
    time_precision    TEXT NOT NULL CHECK (time_precision IN ('part_of_day', 'day')),
    source            TEXT NOT NULL CHECK (source IN ('questionnaire', 'free_text')),
    method            TEXT NOT NULL, -- questionnaire | rules | llm:<model>
    evidence          TEXT,          -- exact words from the report
    confidence        REAL,
    review_status     TEXT NOT NULL CHECK (review_status IN
                        ('auto_accepted', 'needs_review', 'confirmed', 'corrected', 'rejected')),
    review_reason     TEXT
);
CREATE INDEX idx_obs_patient_date ON symptom_observations (patient_id, occurred_date, symptom);

-- Patient-reported medication behaviour (took / missed / helped), distinct from prescriptions.
CREATE TABLE medication_mentions (
    mention_id   INTEGER PRIMARY KEY,
    report_id    TEXT NOT NULL REFERENCES reports(report_id),
    patient_id   TEXT NOT NULL REFERENCES patients(patient_id),
    med_code     TEXT NOT NULL REFERENCES medications(med_code),
    action       TEXT NOT NULL,
    evidence     TEXT,
    method       TEXT NOT NULL
);

CREATE TABLE measurements (
    measurement_id  INTEGER PRIMARY KEY,
    report_id       TEXT REFERENCES reports(report_id),
    patient_id      TEXT NOT NULL REFERENCES patients(patient_id),
    measured_at     TIMESTAMP NOT NULL,
    type            TEXT NOT NULL CHECK (type IN ('weight_lb')),
    value           REAL NOT NULL
);

-- Audit trail of every parser run, including failures and raw model output.
CREATE TABLE extraction_runs (
    run_id       INTEGER PRIMARY KEY,
    report_id    TEXT NOT NULL REFERENCES reports(report_id),
    method       TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
    raw_output   TEXT,
    error        TEXT,
    latency_s    REAL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Items the parser was not allowed to keep (e.g. evidence not found in the text).
CREATE TABLE rejected_extractions (
    rejected_id  INTEGER PRIMARY KEY,
    report_id    TEXT NOT NULL REFERENCES reports(report_id),
    payload      TEXT NOT NULL,
    reason       TEXT NOT NULL
);

-- Daily rollup the dashboard reads. Rebuilt from observations; never edited by hand.
CREATE TABLE daily_symptoms (
    patient_id      TEXT NOT NULL,
    day_date        DATE NOT NULL,
    symptom         TEXT NOT NULL,
    display_value   REAL,        -- questionnaire score / count / 0-1 from text (see value_source)
    display_frac    REAL,        -- 0-1, for colour only
    value_source    TEXT,        -- questionnaire | free_text | both
    text_intensity  TEXT,
    conflict        INTEGER NOT NULL DEFAULT 0,
    conflict_note   TEXT,
    PRIMARY KEY (patient_id, day_date, symptom)
);

-- Observations a person should look at before they count.
CREATE VIEW review_queue AS
SELECT o.obs_id, o.patient_id, r.submitted_at, o.symptom, o.present, o.intensity, o.score,
       o.episode_count, o.evidence, o.review_reason, o.method, r.raw_text
FROM symptom_observations o JOIN reports r USING (report_id)
WHERE o.review_status = 'needs_review';
