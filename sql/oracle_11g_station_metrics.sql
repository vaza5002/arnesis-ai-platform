CREATE TABLE arn_system_setting (
    setting_key VARCHAR2(100) NOT NULL,
    setting_value VARCHAR2(1000) NOT NULL,
    description VARCHAR2(500),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    CONSTRAINT pk_arn_system_setting PRIMARY KEY (setting_key)
);

CREATE TABLE arn_station_metric (
    id NUMBER(19) NOT NULL,
    observed_at TIMESTAMP NOT NULL,
    group_id NUMBER(19) NOT NULL,
    group_code VARCHAR2(50) NOT NULL,
    camera_id NUMBER(19) NOT NULL,
    camera_name VARCHAR2(150) NOT NULL,
    roi_id NUMBER(19) NOT NULL,
    roi_name VARCHAR2(150) NOT NULL,
    head_count NUMBER(10) DEFAULT 0 NOT NULL,
    person_count NUMBER(10) DEFAULT 0 NOT NULL,
    occupied NUMBER(1) DEFAULT 0 NOT NULL,
    va_count NUMBER(10) DEFAULT 0 NOT NULL,
    nva_count NUMBER(10) DEFAULT 0 NOT NULL,
    neutral_count NUMBER(10) DEFAULT 0 NOT NULL,
    cuda_device VARCHAR2(100),
    source_sequence NUMBER(19),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    CONSTRAINT pk_arn_station_metric PRIMARY KEY (id),
    CONSTRAINT uq_arn_metric_obs UNIQUE (group_id, camera_id, roi_id, observed_at),
    CONSTRAINT ck_arn_metric_occupied CHECK (occupied IN (0, 1)),
    CONSTRAINT ck_arn_metric_counts CHECK (
        head_count >= 0 AND person_count >= 0 AND va_count >= 0
        AND nva_count >= 0 AND neutral_count >= 0
    )
);

CREATE SEQUENCE seq_arn_station_metric START WITH 1 INCREMENT BY 1 NOCACHE;
CREATE INDEX ix_arn_metric_observed ON arn_station_metric (observed_at);
CREATE INDEX ix_arn_metric_roi ON arn_station_metric (roi_id);
