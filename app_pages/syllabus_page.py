"""🗺️ Syllabus: units and topics, and which of the student's slides cover each one."""
import streamlit as st

from edu_rag import syllabus

from . import common


def page():
    common.page_start("🗺️", "Syllabus coverage", "Your syllabus topics, and the slides that cover each one. Questions in the chat are also tagged with their unit.")
    sylls = [s for s in common.sources() if s["role"] == "syllabus"]
    if not sylls:
        st.info("Upload your syllabus on the **📚 Library** page (choose **Syllabus** as the file type).")
        return
    with st.spinner("Reading the syllabus units..."):
        units = common.active_syllabus_units()
    if not units:
        st.warning("No units could be read from the syllabus. Is it switched on in the Library?")
        return
    if st.button("Check coverage against my materials", type="primary") or "coverage" in st.session_state:
        if "coverage" not in st.session_state:
            bar = st.progress(0.0, text="Checking topics...")
            st.session_state.coverage = syllabus.coverage(units, sources=common.evidence_filter(),
                                                          progress=lambda i, n: bar.progress(i / n, text=f"Topic {i} of {n}"))
            bar.empty()
        cov = st.session_state.coverage
        gaps = [c for c in cov if not c["covered"]]
        c1, c2 = st.columns(2)
        c1.metric("Topics covered by your materials", f"{len(cov) - len(gaps)} / {len(cov)}")
        c2.metric("Gaps", len(gaps))
        if st.button("↺ Re-check"):
            st.session_state.pop("coverage")
            st.rerun()
    else:
        cov = None
    for u in units:
        with st.container(border=True):
            st.markdown(f"**{u['name']}**")
            for t in u["topics"]:
                row = next((c for c in cov if c["unit"] == u["name"] and c["topic"] == t), None) if cov else None
                if row is None:
                    st.markdown(f"- {t}")
                elif row["covered"]:
                    where = "; ".join(f"{common.short_name(s['source'], 30)} {('p.' if s['unit'] == 'p.' else s['unit'])} {s['pages']}"
                                      for s in row["slides"][:2])
                    st.markdown(f"- ✅ {t}  \n  <span class='small-muted'>{where}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"- ⚠️ **{t}**  \n  <span class='small-muted'>not in your materials: add notes, or ask and use Go deeper</span>",
                                unsafe_allow_html=True)
