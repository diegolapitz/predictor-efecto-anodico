import duckdb
import pandas as pd
from src.semiturn import semiturn_available_at, semiturn_end_case


def test_all_six_closes_match_literal_business_oracle():
    expected = pd.to_datetime(['2026-05-01 10:00', '2026-05-01 14:00',
        '2026-05-01 18:00', '2026-05-01 22:00', '2026-05-02 02:00', '2026-05-02 06:00'])
    actual = semiturn_available_at(['2026-05-01']*6, range(1,7))
    pd.testing.assert_index_equal(actual, expected)
    con=duckdb.connect()
    result=con.execute(f"SELECT TIMESTAMP '2026-05-01' + {semiturn_end_case('s')} t FROM range(1,7) st(s)").fetchdf().t
    assert list(result)==list(expected)
    con.close()


def test_asof_uses_previous_operational_date_through_morning():
    # Literal expected dates are independent of the production mapping.
    v=pd.DataFrame([(pd.Timestamp(d),s) for d in ['2026-04-30','2026-05-01','2026-05-02'] for s in range(1,7)],columns=['ft_turno','st'])
    v['cuba']='1001'
    h=pd.DataFrame({'timestamp':pd.date_range('2026-05-01 05:55','2026-05-02 10:05',freq='5min'),'cuba':'1001'})
    con=duckdb.connect();con.register('v',v);con.register('h',h)
    result=con.execute(f'''SELECT h.timestamp,v.ft_turno,v.st,v.available_at FROM h
        ASOF LEFT JOIN (SELECT *,ft_turno+{semiturn_end_case('st')} available_at FROM v) v
        ON h.cuba=v.cuba AND h.timestamp>=v.available_at ORDER BY h.timestamp''').fetchdf()
    for day in ['2026-05-01','2026-05-02']:
        d=pd.Timestamp(day)
        early=result[result.timestamp.eq(d+pd.Timedelta(hours=5,minutes=55))].iloc[0]
        assert early.st==5 and early.ft_turno==d-pd.Timedelta(days=1)
        band=result[result.timestamp.ge(d+pd.Timedelta(hours=6)) & result.timestamp.lt(d+pd.Timedelta(hours=10))]
        assert len(band)==48 and band.st.eq(6).all()
        assert band.ft_turno.eq(d-pd.Timedelta(days=1)).all()
        assert band.available_at.eq(d+pd.Timedelta(hours=6)).all()
        at_ten=result[result.timestamp.eq(d+pd.Timedelta(hours=10))].iloc[0]
        assert at_ten.st==1 and at_ten.ft_turno==d
    con.close()
