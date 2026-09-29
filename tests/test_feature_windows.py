import duckdb


def test_time_window_never_uses_future_row():
    con = duckdb.connect()
    result = con.execute(
        """
        WITH sample(cuba, ts, value) AS (
            VALUES
                ('501', TIMESTAMP '2026-01-01 00:00:00', 1.0),
                ('501', TIMESTAMP '2026-01-01 00:05:00', 2.0),
                ('501', TIMESTAMP '2026-01-01 00:10:00', 100.0)
        )
        SELECT ts,
               avg(value) OVER (
                   PARTITION BY cuba ORDER BY ts
                   RANGE BETWEEN INTERVAL 10 MINUTE PRECEDING AND CURRENT ROW
               ) AS trailing_mean
        FROM sample
        ORDER BY ts
        """
    ).fetchall()
    assert result[0][1] == 1.0
    assert result[1][1] == 1.5
    assert result[2][1] == (1.0 + 2.0 + 100.0) / 3.0

