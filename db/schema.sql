CREATE TABLE IF NOT EXISTS storage(
    idx SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    original_answer TEXT NOT NULL,
    candidate_answers JSONB
);