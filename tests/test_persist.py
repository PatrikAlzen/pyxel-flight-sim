import json

import pytest

from flightsim.persist import MAX_PLACES, VERSION, SaveStore, WorldSave


@pytest.fixture
def store(tmp_path):
    return SaveStore(tmp_path / "saves")


def test_missing_file_gives_an_empty_world(store):
    world = store.load(42)
    assert world == WorldSave(42)


def test_round_trip(store):
    world = WorldSave(7, wrecks=[[1.0, 2.0, 3.0]])
    world.add_place(10.0, -20.0, 300.0, 1.5)
    store.save(world)
    loaded = store.load(7)
    assert loaded == world
    assert loaded.places[0].name == "PLACE 1"
    assert json.loads(store.path(7).read_text())["version"] == VERSION


def test_saving_leaves_no_temporary_file(store):
    store.save(WorldSave(1))
    assert [p.name for p in store.directory.iterdir()] == ["world-1.json"]


def test_older_file_without_new_fields_still_loads(store):
    store.directory.mkdir(parents=True)
    store.path(5).write_text(
        json.dumps({"wrecks": [[4, 5, 6]]})
    )  # no version, no places
    world = store.load(5)
    assert world.wrecks == [[4.0, 5.0, 6.0]]
    assert world.places == []


def test_unknown_fields_are_ignored(store):
    store.directory.mkdir(parents=True)
    future = {
        "version": VERSION + 3,
        "wrecks": [],
        "places": [
            {"name": "A", "x": 1, "y": 2, "altitude": 3, "heading": 0, "colour": "red"}
        ],
        "lap_records": {"x": 1},
    }
    store.path(5).write_text(json.dumps(future))
    assert store.load(5).places[0].name == "A"


@pytest.mark.parametrize("content", ["{not json", '{"wrecks": "oops"}', '["a list"]'])
def test_corrupt_file_is_moved_aside_and_world_starts_empty(store, content):
    store.directory.mkdir(parents=True)
    store.path(9).write_text(content)
    assert store.load(9) == WorldSave(9)
    assert not store.path(9).exists()
    assert store.path(9).with_suffix(".json.corrupt").read_text() == content


def test_place_names_reuse_free_numbers_and_are_capped():
    world = WorldSave(1)
    for i in range(3):
        world.add_place(i, 0, 100, 0)
    del world.places[1]  # PLACE 2 removed
    assert world.add_place(9, 9, 9, 9).name == "PLACE 2"
    for i in range(MAX_PLACES + 5):
        world.add_place(i, 0, 0, 0)
    assert len(world.places) == MAX_PLACES


def test_summary_counts_without_loading(store):
    world = WorldSave(3, wrecks=[[0, 0, 0], [1, 1, 1]])
    world.add_place(0, 0, 0, 0)
    store.save(world)
    assert store.summary(3) == (2, 1)
    assert store.summary(4) == (0, 0)


def test_visited_airfields_round_trip(store):
    world = WorldSave(11, visited=[389, 512])
    store.save(world)
    assert store.load(11).visited == [389, 512]


def test_time_trial_records_round_trip(store):
    world = WorldSave(
        12, records={657: {"splits": [10.5, 20.25], "ghost": [[1, 2, 3]]}}
    )
    store.save(world)
    loaded = store.load(12)
    assert loaded.records == {
        657: {"splits": [10.5, 20.25], "ghost": [[1.0, 2.0, 3.0]]}
    }
