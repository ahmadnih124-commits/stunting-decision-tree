import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Decision Tree Risiko Stunting", page_icon="🌳", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 20%, rgba(255,255,255,0.55) 0, transparent 35%),
            radial-gradient(circle at 85% 80%, rgba(255,255,255,0.40) 0, transparent 40%),
            linear-gradient(135deg, #d9f2e6 0%, #cfe8f7 50%, #e3dcf7 100%);
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        background: rgba(255, 255, 255, 0.82);
        backdrop-filter: blur(6px);
        border-radius: 20px;
        padding: 2rem 2.5rem 3rem 2.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(40, 80, 120, 0.15);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #b8e0d2 0%, #a9cfe8 100%);
    }
    h1 { color: #1b4d3e; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

TARGET = "Risiko stunting"
PREDIKTOR = ["Usia (bulan)", "Asupan gizi", "Riwayat infeksi", "Pertumbuhan"]

DATA_AWAL = pd.DataFrame(
    [
        ["18–24", "Baik", "Tidak", "Normal", "Rendah"],
        ["18–24", "Kurang", "Ya", "Terhambat", "Tinggi"],
        ["25–36", "Baik", "Tidak", "Normal", "Rendah"],
        ["25–36", "Kurang", "Ya", "Terhambat", "Tinggi"],
        ["18–24", "Kurang", "Tidak", "Terhambat", "Tinggi"],
        ["37–59", "Baik", "Ya", "Normal", "Rendah"],
        ["37–59", "Kurang", "Tidak", "Terhambat", "Tinggi"],
        ["25–36", "Baik", "Ya", "Normal", "Rendah"],
        ["18–24", "Baik", "Ya", "Terhambat", "Tinggi"],
        ["37–59", "Baik", "Tidak", "Normal", "Rendah"],
    ],
    columns=PREDIKTOR + [TARGET],
)
DATA_AWAL.index = range(1, len(DATA_AWAL) + 1)
DATA_AWAL.index.name = "No."


# ---------- Fungsi inti Decision Tree (ID3) ----------
def entropy(series):
    n = len(series)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in series.value_counts() if c > 0)


def info_gain(df, attr):
    total = entropy(df[TARGET])
    weighted = sum(len(sub) / len(df) * entropy(sub[TARGET]) for _, sub in df.groupby(attr))
    return total - weighted


def build_tree(df, attrs):
    majority = df[TARGET].mode()[0]
    if df[TARGET].nunique() == 1:
        return {"leaf": majority, "n": len(df)}
    if not attrs:
        return {"leaf": majority, "n": len(df)}
    gains = {a: info_gain(df, a) for a in attrs}
    best = max(gains, key=gains.get)
    if gains[best] <= 0:
        return {"leaf": majority, "n": len(df)}
    rest = [a for a in attrs if a != best]
    return {
        "attr": best,
        "majority": majority,
        "children": {v: build_tree(sub, rest) for v, sub in df.groupby(best)},
    }


def classify(tree, sample):
    jalur = []
    while "leaf" not in tree:
        val = sample[tree["attr"]]
        jalur.append(f"{tree['attr']} = {val}")
        if val not in tree["children"]:
            return tree["majority"], jalur
        tree = tree["children"][val]
    return tree["leaf"], jalur


def rules(tree, kondisi=None):
    kondisi = kondisi or []
    if "leaf" in tree:
        kata = " AND ".join(kondisi) if kondisi else "(semua data)"
        return [f"IF {kata} THEN {TARGET} = {tree['leaf']}"]
    out = []
    for v, child in tree["children"].items():
        out += rules(child, kondisi + [f"{tree['attr']} = {v}"])
    return out


def to_dot(tree):
    lines = ["digraph T {", 'node [fontname="Helvetica"];']
    counter = [0]

    def walk(node):
        my = counter[0]
        counter[0] += 1
        if "leaf" in node:
            warna = "#f8b4b4" if node["leaf"] == "Tinggi" else "#b7e4c7"
            lines.append(f'n{my} [label="{TARGET}:\\n{node["leaf"]}\\n(n={node["n"]})", shape=box, style="rounded,filled", fillcolor="{warna}"];')
        else:
            lines.append(f'n{my} [label="{node["attr"]}", shape=ellipse, style=filled, fillcolor="#cfe2ff"];')
            for v, child in node["children"].items():
                cid = walk(child)
                lines.append(f'n{my} -> n{cid} [label="{v}"];')
        return my

    walk(tree)
    lines.append("}")
    return "\n".join(lines)


# ---------- Tampilan ----------
st.title("🌳 Klasifikasi Risiko Stunting Balita – Decision Tree")
st.caption("Dataset ilustratif untuk latihan pembelajaran, bukan data pasien nyata dan bukan alat diagnosis klinis.")

with st.sidebar:
    st.header("Data")
    st.write("Dataset bawaan = 10 record dari studi kasus. Kamu bisa mengubahnya di tab **1**; seluruh perhitungan ikut berubah otomatis.")
    if st.button("Reset ke dataset awal"):
        st.session_state.pop("editor", None)
        st.session_state["data"] = DATA_AWAL.copy()
        st.rerun()

if "data" not in st.session_state:
    st.session_state["data"] = DATA_AWAL.copy()

tabs = st.tabs([
    "1. Atribut & Data", "2. Distribusi Kelas", "3. Entropy", "4. Information Gain",
    "5. Root Node", "6. Pohon & Aturan IF–THEN", "7. Klasifikasi Data Baru", "8. Keterbatasan",
])

# Tab 1
with tabs[0]:
    st.subheader("Atribut prediktor dan atribut target")
    c1, c2 = st.columns(2)
    c1.markdown("**Atribut prediktor (X):**\n" + "\n".join(f"- {a}" for a in PREDIKTOR))
    c2.markdown(f"**Atribut target (Y):**\n- {TARGET}")
    st.markdown("Dataset (bisa diedit):")
    edited = st.data_editor(st.session_state["data"], key="editor", num_rows="dynamic", use_container_width=True)
    st.session_state["data"] = edited

df = st.session_state["data"].dropna()
if len(df) < 2 or df[TARGET].nunique() < 1:
    st.warning("Data terlalu sedikit untuk dihitung.")
    st.stop()

# Tab 2
with tabs[1]:
    st.subheader("Jumlah data pada setiap kelas target")
    cnt = df[TARGET].value_counts()
    tabel = pd.DataFrame({"Jumlah": cnt, "Persentase (%)": (cnt / len(df) * 100).round(1)})
    st.dataframe(tabel)
    st.bar_chart(cnt)
    selisih = cnt.max() - cnt.min()
    if selisih == 0 or cnt.min() / cnt.max() >= 0.8:
        st.success(f"Distribusi kelas **seimbang** ({', '.join(f'{k}: {v}' for k, v in cnt.items())}).")
    else:
        st.error("Distribusi kelas **tidak seimbang**.")

# Tab 3
with tabs[2]:
    st.subheader("Entropy seluruh dataset")
    n = len(df)
    st.latex(r"Entropy(S) = -\sum_i p_i \log_2 p_i")
    suku = " + ".join(f"\\left(\\frac{{{c}}}{{{n}}}\\log_2\\frac{{{c}}}{{{n}}}\\right)" for c in cnt)
    st.latex(f"Entropy(S) = -\\left[{suku}\\right]")
    H = entropy(df[TARGET])
    st.metric("Entropy(S)", f"{H:.4f}")

# Tab 4
with tabs[3]:
    st.subheader("Information Gain tiap atribut")
    st.latex(r"Gain(S,A) = Entropy(S) - \sum_v \frac{|S_v|}{|S|} Entropy(S_v)")
    gains = {a: info_gain(df, a) for a in PREDIKTOR}
    pilihan = st.multiselect("Atribut yang ditampilkan rinciannya", PREDIKTOR, default=["Asupan gizi", "Pertumbuhan"])
    for a in pilihan:
        st.markdown(f"**{a}**")
        baris = []
        for v, sub in df.groupby(a):
            baris.append({
                a: v, "Jumlah": len(sub),
                **{f"n {k}": int((sub[TARGET] == k).sum()) for k in cnt.index},
                "Entropy": round(entropy(sub[TARGET]), 4),
            })
        st.dataframe(pd.DataFrame(baris), hide_index=True)
        st.write(f"Gain({a}) = **{gains[a]:.4f}**")
    st.markdown("**Ringkasan semua atribut:**")
    ring = pd.DataFrame({"Information Gain": gains}).sort_values("Information Gain", ascending=False).round(4)
    st.dataframe(ring)
    st.bar_chart(ring)

# Tab 5
with tabs[4]:
    st.subheader("Atribut terbaik untuk root node")
    best = max(gains, key=gains.get)
    st.success(f"Root node = **{best}** (Information Gain tertinggi = {gains[best]:.4f})")
    st.write("Atribut dengan Information Gain terbesar paling efektif mengurangi ketidakpastian (entropy), sehingga dipilih sebagai akar pohon.")

tree = build_tree(df, PREDIKTOR)

# Tab 6
with tabs[5]:
    st.subheader("Pohon keputusan")
    st.graphviz_chart(to_dot(tree))
    st.subheader("Aturan klasifikasi IF–THEN")
    for i, r in enumerate(rules(tree), 1):
        st.code(f"R{i}: {r}", language=None)

# Tab 7
with tabs[6]:
    st.subheader("Klasifikasikan balita baru")
    st.write("Nilai bawaan = kasus dari soal nomor 7. Ubah sesuai kebutuhan lalu klik tombol.")
    opsi = {a: sorted(df[a].unique()) for a in PREDIKTOR}
    default = {"Usia (bulan)": "25–36", "Asupan gizi": "Kurang", "Riwayat infeksi": "Tidak", "Pertumbuhan": "Terhambat"}
    cols = st.columns(4)
    sample = {}
    for col, a in zip(cols, PREDIKTOR):
        idx = opsi[a].index(default[a]) if default[a] in opsi[a] else 0
        sample[a] = col.selectbox(a, opsi[a], index=idx)
    if st.button("Klasifikasikan", type="primary"):
        hasil, jalur = classify(tree, sample)
        st.markdown("**Jalur keputusan:** " + " → ".join(jalur) + f" → **{hasil}**")
        if hasil == "Tinggi":
            st.error(f"Prediksi {TARGET}: **{hasil}**")
        else:
            st.success(f"Prediksi {TARGET}: **{hasil}**")

# Tab 8
with tabs[7]:
    st.subheader("Keterbatasan dataset berukuran kecil")
    st.markdown("""
- **Overfitting**: pohon mudah "menghafal" 10 record, bukan mempelajari pola umum, sehingga akurasi pada data baru bisa buruk.
- **Tidak representatif**: 10 balita tidak mewakili variasi usia, kondisi sosial-ekonomi, wilayah, dan kombinasi faktor risiko yang sebenarnya.
- **Pemisahan terlalu sempurna**: atribut Pertumbuhan memisahkan kelas 100%, kemungkinan karena kebetulan dan datanya ilustratif. Pada data nyata, pertumbuhan terhambat adalah gejala/hasil stunting, bukan sekadar prediktor.
- **Tidak bisa divalidasi andal**: data terlalu sedikit untuk train-test split atau cross-validation yang bermakna, sehingga performa tidak dapat diukur dengan layak.
- **Tidak stabil**: menambah/mengubah satu record bisa mengubah struktur pohon secara drastis.
- **Kombinasi nilai tidak tercakup**: banyak kombinasi atribut tidak muncul di data sehingga model tidak punya dasar untuk memprediksinya.
- **Atribut kurang lengkap**: faktor penting seperti tinggi badan menurut umur (z-score), ASI eksklusif, sanitasi, dan status ekonomi tidak ada.
- **Risiko etis**: model ini hanya untuk pembelajaran, tidak boleh dipakai sebagai alat diagnosis klinis.
""")
