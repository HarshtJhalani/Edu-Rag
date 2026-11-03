"""Syllabus: semesters, courses, units and topics, and which of the student's slides cover each topic."""
import streamlit as st

from edu_rag import syllabus

from . import common


def topic_line(row, topic):
    if row is None:
        return f"- {topic}"
    if row["covered"]:
        where = "; ".join(f"{common.short_name(s['source'], 30)} {('p.' if s['unit'] == 'p.' else s['unit'])} {s['pages']}"
                          for s in row["slides"][:2])
        return f"- ✅ {topic}  \n  <span class='small-muted'>{where}</span>"
    return f"- ⚠️ **{topic}**  \n  <span class='small-muted'>not in this course's files: add notes, or ask and use Go deeper</span>"


def topic_lines(unit, by_topic):
    """One element per unit: 846 of them made every click slow."""
    return "\n".join(topic_line(by_topic.get((unit["name"], t)), t) for t in unit["topics"])


def course_files_box(course, course_map, files):
    """The files this course's topics are searched in. Estimated from their content, so the student can correct it."""
    chosen = [f for f in course_map.get(course, []) if f in files]
    picked = st.multiselect("Files for this course", list(files), default=chosen, format_func=files.get, key=f"files_{course}",
                            help="Topics of this course are only looked for in these files. Chosen automatically: correct them if needed.")
    if set(picked) != set(chosen):
        saved = dict(common.pref("course_files", {}))
        saved[course] = picked
        common.set_pref("course_files", saved)
        st.session_state.pop("coverage", None)
        st.rerun()


def course_card(rows, by_topic, course_map, files):
    """One course: its units and topics, closed by default because a programme has many courses."""
    first = rows[0]
    n = sum(len(u["topics"]) for u in rows)
    title = " · ".join(x for x in (first["code"], first["course"]) if x)
    with st.expander(f"**{title}** ({n} topics)" if n else f"**{title}** (no units or topics listed)"):
        if n and first["course"]:
            course_files_box(first["course"], course_map, files)
        for u in rows:
            if u["unit"]:
                st.markdown(f"**{u['unit']}**")
            if u["topics"]:
                st.markdown(topic_lines(u, by_topic), unsafe_allow_html=True)
        st.caption(f"Syllabus PDF, p. {first['pages'][0]}-{first['pages'][-1]}")


def show_units(units, cov, course_map):
    by_topic = {(c["unit"], c["topic"]): c for c in cov or []}
    files = {s["source_id"]: s["name"] for s in common.evidence_sources()}      # read once: each read goes through every source's folder
    for semester in syllabus.semesters(units) or [None]:
        if semester:
            st.markdown(f"#### {semester}")
        rows = [u for u in units if u.get("semester") == semester]
        courses = {}
        for u in rows:
            courses.setdefault((u.get("pages") or [0])[0] if u.get("course") or u.get("code") else "", []).append(u)
        for name, group in courses.items():
            if group[0].get("course") or group[0].get("code"):
                course_card(group, by_topic, course_map, files)
            else:                                   # a syllabus without course pages: plain units
                for u in group:
                    with st.container(border=True):
                        st.markdown(f"**{u['name']}**")
                        if u["topics"]:
                            st.markdown(topic_lines(u, by_topic), unsafe_allow_html=True)


def page():
    common.page_start("Syllabus coverage", "Your syllabus topics, and the slides that cover each one. Questions in the chat are also tagged with their unit.")
    sylls = [s for s in common.sources() if s["role"] == "syllabus"]
    if not sylls:
        st.info("Upload your syllabus on the **Library** page (choose **Syllabus** as the file type).")
        return
    with st.spinner("Reading the syllabus, course by course..."):
        common.semester_picker("syllabus_semester")
        units = common.active_syllabus_units()
    if not units:
        st.warning("No units could be read from the syllabus. Is it switched on in the Library?")
        return
    common.stats_row([(len({(u.get("pages") or [0])[0] for u in units}), "courses"), (sum(bool(u["topics"]) for u in units), "units"),
                      (sum(len(u["topics"]) for u in units), "topics")])
    names = [u["name"] for u in units]
    with st.spinner("Matching your files to the courses..."):
        course_map = common.course_file_map()
    if st.session_state.get("coverage_for") != names:      # another semester: the old result does not apply
        st.session_state.pop("coverage", None)
    if st.button("Check coverage against my materials", type="primary") or "coverage" in st.session_state:
        if "coverage" not in st.session_state:
            bar = st.progress(0.0, text="Checking topics...")
            st.session_state.coverage = syllabus.coverage(units, sources=common.evidence_filter(),
                                                          progress=lambda i, n: bar.progress(i / n, text=f"Topic {i} of {n}"),
                                                          course_map=course_map)
            st.session_state.coverage_for = names
            bar.empty()
        cov = st.session_state.coverage
        gaps = [c for c in cov if not c["covered"]]
        c1, c2 = st.columns(2)
        c1.metric("Topics covered by your materials", f"{len(cov) - len(gaps)} / {len(cov)}")
        c2.metric("Gaps", len(gaps))
        if st.button("Re-check"):
            st.session_state.pop("coverage")
            st.rerun()
    else:
        cov = None
    show_units(units, cov, course_map)
