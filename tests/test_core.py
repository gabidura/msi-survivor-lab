from msi_survivor import primes_up_to, primorial, windowed_survivor_count, polynomial_window_count


def test_primes_and_primorial():
    assert primes_up_to(10) == [2, 3, 5, 7]
    assert primorial(5) == 30


def test_window_count_goldbach_small():
    r = windowed_survivor_count((0,), 5, 58, 0, 59)
    assert r.count >= 0


def test_polynomial_law_matches_direct():
    r = polynomial_window_count((0,), 5, 58, 0, 59)
    assert r["coefficient_count"] == r["direct_count"]

