from edu_rag import library


def doc(sid, name, role="textbook", **kw):
    return dict(source_id=sid, name=name, role=role, n_pages=1, hash=sid * 3, pages=[], **kw)


def test_safe_name():
    assert library.safe_name("Ch 3. Classification.pdf") == "Ch 3. Classification"
    assert library.safe_name("coursera.org: What Is an Entity?") == "coursera.org What Is an Entity"
    assert library.safe_name("Meow laadle ") == "Meow laadle"


def test_one_folder_per_source_with_its_file(tmp_path):
    folder = library.save_source(doc("aaa111", "Deck.pdf", file="Deck.pdf"), {"Deck.pdf": b"%PDF"}, root=tmp_path)
    assert folder == tmp_path / "textbook" / "Deck"
    assert (folder / "Deck.pdf").read_bytes() == b"%PDF" and (folder / "source.json").exists()
    assert library.load_source("aaa111", tmp_path)["name"] == "Deck.pdf"
    assert library.original_file("aaa111", tmp_path) == folder / "Deck.pdf"
    assert [s["source_id"] for s in library.list_sources(tmp_path)] == ["aaa111"]


def test_same_name_different_file_gets_its_own_folder(tmp_path):
    a = library.save_source(doc("aaa111", "Deck.pdf"), root=tmp_path)
    b = library.save_source(doc("bbb222", "Deck.pdf"), root=tmp_path)
    assert a != b and b.name == "Deck (bbb222)"
    assert library.save_source(doc("aaa111", "Deck.pdf"), root=tmp_path) == a        # re-saving reuses the folder


def test_delete_removes_only_that_folder(tmp_path):
    library.save_source(doc("aaa111", "A.pdf"), root=tmp_path)
    library.save_source(doc("bbb222", "B note", role="note"), root=tmp_path)
    assert library.delete_source("aaa111", tmp_path)
    assert [s["source_id"] for s in library.list_sources(tmp_path)] == ["bbb222"]
    assert not library.delete_source("missing", tmp_path)
