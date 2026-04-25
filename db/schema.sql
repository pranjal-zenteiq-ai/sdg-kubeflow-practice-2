CREATE TABLE IF NOT EXISTS storage2(
    idx SERIAL PRIMARY KEY,
    stage TEXT NOT NULL,
    question TEXT NOT NULL,
    original_answer TEXT,
    payload JSONB NOT NULL
);