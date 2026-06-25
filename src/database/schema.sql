CREATE TABLE IF NOT EXISTS station_information (
    station_id TEXT PRIMARY KEY,
    station_name TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    capacity DOUBLE PRECISION,
    first_seen_at_utc TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at_utc TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS station_status_snapshots (
    id BIGSERIAL PRIMARY KEY,
    snapshot_utc TIMESTAMPTZ NOT NULL,
    station_id TEXT NOT NULL,
    num_bikes_available DOUBLE PRECISION,
    num_ebikes_available DOUBLE PRECISION,
    num_docks_available DOUBLE PRECISION,
    pct_bikes_available DOUBLE PRECISION,
    pct_docks_available DOUBLE PRECISION,
    availability_status TEXT,
    is_installed INTEGER,
    is_renting INTEGER,
    is_returning INTEGER,
    last_reported_utc TIMESTAMPTZ,
    status_fetched_at_utc TIMESTAMPTZ,
    inserted_at_utc TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (snapshot_utc, station_id)
);

CREATE INDEX IF NOT EXISTS idx_station_status_snapshots_station_time
ON station_status_snapshots (station_id, snapshot_utc DESC);

CREATE INDEX IF NOT EXISTS idx_station_status_snapshots_time
ON station_status_snapshots (snapshot_utc DESC);