from app.eval.coverage import locate_point, neighbours_could_answer


def test_a_point_in_the_context_is_found_there_first() -> None:
    assert locate_point("1,6 mm", "Profiltiefe von 1.6 mm", "auch 1,6 mm") == "context"


def test_a_point_only_in_a_neighbour() -> None:
    assert locate_point("ein Unterlegkeil", "Feststellbremse", "Ein  Unterlegkeil") == "neighbour"


def test_a_point_nowhere() -> None:
    assert locate_point("5,0 m/s2", "Bremsen", "Anhaenger") == "missing"


def test_neighbours_help_only_when_they_complete_the_context() -> None:
    assert neighbours_could_answer(["context", "neighbour"])
    assert not neighbours_could_answer(["context", "context"])
    assert not neighbours_could_answer(["neighbour", "missing"])
    assert not neighbours_could_answer([])
