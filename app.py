import io
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from Bio.PDB import MMCIFParser, PDBParser


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="Protein Insight",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7faff;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .header-title {
        font-size: 32px;
        font-weight: 800;
        color: #123b78;
        margin-bottom: 0;
    }

    .header-subtitle {
        color: #66758a;
        font-size: 14px;
        margin-top: 4px;
    }

    .card {
        background-color: white;
        border-radius: 14px;
        padding: 22px;
        border: 1px solid #e3eaf3;
        box-shadow: 0 2px 8px rgba(40, 70, 110, 0.06);
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #173b70;
        margin-bottom: 12px;
    }

    .description {
        color: #66758a;
        line-height: 1.7;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e3eaf3;
        padding: 15px;
        border-radius: 12px;
    }

    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 함수
# =========================================================

def parse_structure(file_name, file_bytes):
    """PDB/mmCIF 파일에서 CA 원자의 B-factor를 pLDDT로 추출."""
    text_stream = io.StringIO(file_bytes.decode("utf-8"))

    if file_name.lower().endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)

    structure = parser.get_structure("protein", text_stream)

    data = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" not in residue:
                    continue

                ca = residue["CA"]

                data.append(
                    {
                        "chain": chain.id,
                        "residue": residue.resname,
                        "position": residue.id[1],
                        "pLDDT": float(ca.get_bfactor()),
                    }
                )

    return pd.DataFrame(data)


def parse_pae(file_bytes):
    """AlphaFold PAE JSON에서 PAE 행렬을 추출."""
    pae_data = json.loads(file_bytes.decode("utf-8"))

    if isinstance(pae_data, list):
        pae_obj = pae_data[0]
    elif isinstance(pae_data, dict):
        pae_obj = pae_data
    else:
        raise ValueError("지원하지 않는 PAE JSON 형식입니다.")

    pae = np.asarray(
        pae_obj["predicted_aligned_error"],
        dtype=float,
    )

    max_pae = pae_obj.get(
        "max_predicted_aligned_error",
        float(np.max(pae)),
    )

    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError("PAE 데이터가 정사각 행렬이 아닙니다.")

    return pae, float(max_pae)


def make_3d_viewer(structure_text, file_name):
    """구조 파일을 py3Dmol viewer로 불러옵니다."""
    viewer = py3Dmol.view(width=900, height=650)

    if file_name.lower().endswith(".pdb"):
        viewer.addModel(structure_text, "pdb")
    else:
        viewer.addModel(structure_text, "mmcif")

    viewer.setBackgroundColor("white")
    return viewer


def render_plddt_3d(structure_text, file_name, df):
    """잔기별 pLDDT 구간에 따라 3D 구조를 색상으로 표시합니다."""
    viewer = make_3d_viewer(structure_text, file_name)

    # 기본값
    viewer.setStyle(
        {},
        {"cartoon": {"color": "lightgray"}},
    )

    # AlphaFold에서 널리 사용하는 pLDDT 해석 구간
    # 90-100: 매우 높음 / 70-90: 높음 / 50-70: 낮음 / 0-50: 매우 낮음
    categories = [
        ("very_high", "blue", df[df["pLDDT"] >= 90]),
        ("high", "cyan", df[(df["pLDDT"] >= 70) & (df["pLDDT"] < 90)]),
        ("low", "yellow", df[(df["pLDDT"] >= 50) & (df["pLDDT"] < 70)]),
        ("very_low", "red", df[df["pLDDT"] < 50]),
    ]

    for _, color, subset in categories:
        for chain_id, chain_df in subset.groupby("chain"):
            positions = chain_df["position"].astype(int).tolist()
            if not positions:
                continue

            viewer.setStyle(
                {"chain": str(chain_id), "resi": positions},
                {"cartoon": {"color": color}},
            )

    viewer.zoomTo()
    components.html(
        viewer._make_html(),
        height=680,
        scrolling=False,
    )


def render_attention_3d(
    structure_text,
    file_name,
    attention_rows,
):
    """종합 주의 residue를 빨간색으로 강조합니다."""
    viewer = make_3d_viewer(structure_text, file_name)

    viewer.setStyle(
        {},
        {"cartoon": {"color": "lightgray"}},
    )

    if not attention_rows.empty:
        for chain_id, chain_df in attention_rows.groupby("chain"):
            positions = chain_df["position"].astype(int).tolist()

            viewer.setStyle(
                {"chain": str(chain_id), "resi": positions},
                {
                    "cartoon": {"color": "red"},
                    "stick": {"color": "red"},
                },
            )

            for pos in positions:
                viewer.addLabel(
                    str(pos),
                    {
                        "chain": str(chain_id),
                        "resi": int(pos),
                        "backgroundColor": "white",
                        "fontColor": "black",
                        "fontSize": 12,
                    },
                )

    viewer.zoomTo()
    components.html(
        viewer._make_html(),
        height=680,
        scrolling=False,
    )


# =========================================================
# Header
# =========================================================

col_logo, col_title, col_menu = st.columns([1, 5, 2])

with col_logo:
    st.markdown("## 🧬")

with col_title:
    st.markdown(
        '<div class="header-title">Protein Insight</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="header-subtitle">'
        "Understanding Confidence in Protein Structures"
        "</div>",
        unsafe_allow_html=True,
    )

with col_menu:
    st.markdown(
        "<div style='text-align:right; color:#526176; padding-top:15px;'>"
        "Protein Structure Analysis"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.markdown("## 🧬 Protein Insight")
    st.markdown("---")

    page = st.radio(
        "메뉴",
        [
            "🏠 단백질 분석",
            "📊 결과 예시",
            "⚖️ 비교 분석",
            "📁 자료실",
            "ℹ️ 소개",
            "❓ 도움말",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if page == "🏠 단백질 분석":
        st.markdown("### 📤 단백질 파일 업로드")
        st.caption(
            "AlphaFold에서 다운로드한 구조 파일과 PAE JSON을 업로드하세요."
        )

        structure_file = st.file_uploader(
            "PDB / mmCIF",
            type=["pdb", "cif", "mmcif"],
            key="structure",
        )

        pae_file = st.file_uploader(
            "PAE JSON",
            type=["json"],
            key="pae",
        )

        analyze_button = st.button(
            "🔬 분석 시작",
            use_container_width=True,
            type="primary",
        )
    else:
        structure_file = None
        pae_file = None
        analyze_button = False


# =========================================================
# 단백질 분석
# =========================================================

if page == "🏠 단백질 분석":

    st.markdown("## AI가 예측한 구조, 어디까지 믿을 수 있을까?")
    st.markdown(
        '<div class="description">'
        "단백질 구조 예측 결과의 불확실성을 "
        "pLDDT와 PAE를 이용하여 분석하는 도구입니다."
        "</div>",
        unsafe_allow_html=True,
    )

    if not analyze_button:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">🧬 Protein Insight에 오신 것을 환영합니다</div>
                <div class="description">
                    왼쪽 사이드바에서 AlphaFold 구조 파일(PDB/mmCIF)과
                    PAE JSON을 업로드한 뒤 <b>🔬 분석 시작</b>을 눌러주세요.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                <div class="card">
                    <h3>📊 pLDDT</h3>
                    residue별 국소 구조 예측 신뢰도를 확인합니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="card">
                    <h3>🔥 PAE</h3>
                    residue 또는 영역 사이의 상대적 위치
                    예측 불확실성을 확인합니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
                <div class="card">
                    <h3>🧬 3D Structure</h3>
                    pLDDT 신뢰도와
                    종합 주의 영역을 3차원으로 확인합니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

    if analyze_button:

        if structure_file is None or pae_file is None:
            st.warning(
                "구조 파일과 PAE JSON 파일을 모두 업로드해주세요."
            )
            st.stop()

        try:
            # -------------------------
            # 구조 파일 분석
            # -------------------------

            file_name = structure_file.name.lower()
            file_bytes = structure_file.getvalue()

            df = parse_structure(file_name, file_bytes)

            if df.empty:
                st.error(
                    "구조 파일에서 CA 원자를 찾지 못했습니다."
                )
                st.stop()

            # -------------------------
            # PAE 분석
            # -------------------------

            pae, max_pae = parse_pae(
                pae_file.getvalue()
            )

            if pae.shape[0] != len(df):
                st.error(
                    "PAE 데이터의 크기와 구조 파일의 residue 수가 "
                    "일치하지 않습니다."
                )
                st.info(
                    f"구조 파일 residue 수: {len(df)} / "
                    f"PAE 크기: {pae.shape[0]}"
                )
                st.stop()

            # -------------------------
            # pLDDT 분석
            # -------------------------

            mean_plddt = df["pLDDT"].mean()
            min_row = df.loc[df["pLDDT"].idxmin()]
            max_row = df.loc[df["pLDDT"].idxmax()]

            # -------------------------
            # PAE 분석
            # -------------------------

            mean_pae = float(pae.mean())
            mean_pae_by_residue = pae.mean(axis=1)

            pae_df = pd.DataFrame(
                {
                    "position": df["position"].to_numpy(),
                    "mean_PAE": mean_pae_by_residue,
                }
            )

            # -------------------------
            # 종합 주의 영역
            # -------------------------

            # 3D 주의 영역 표시를 위해 chain 정보도 유지합니다.
            merged = df[
                ["chain", "residue", "position", "pLDDT"]
            ].copy()

            merged["mean_PAE"] = mean_pae_by_residue

            pae_cutoff = merged["mean_PAE"].mean()

            merged["attention"] = (
                (merged["pLDDT"] < 70)
                & (merged["mean_PAE"] > pae_cutoff)
            )

            attention_positions = (
                merged.loc[
                    merged["attention"],
                    "position",
                ]
                .tolist()
            )

            # -------------------------
            # 결과 요약
            # -------------------------

            st.success("분석이 완료되었습니다.")

            st.markdown("## 📊 pLDDT 분석 결과")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "단백질 길이",
                    f"{len(df)} residues",
                )

            with col2:
                st.metric(
                    "평균 pLDDT",
                    f"{mean_plddt:.2f}",
                )

            with col3:
                st.metric(
                    "최저 pLDDT",
                    f"{min_row['pLDDT']:.2f}",
                )

            with col4:
                st.metric(
                    "최고 pLDDT",
                    f"{max_row['pLDDT']:.2f}",
                )

            # -------------------------
            # pLDDT 그래프
            # -------------------------

            st.markdown("### 🧬 Residue별 pLDDT")

            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(
                df["position"],
                df["pLDDT"],
            )

            ax.axhline(
                70,
                linestyle="--",
                label="pLDDT = 70",
            )

            ax.set_xlabel("Residue position")
            ax.set_ylabel("pLDDT")
            ax.set_title(
                "pLDDT Confidence by Residue"
            )
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

            # -------------------------
            # PAE 결과
            # -------------------------

            st.markdown("## 📊 PAE 분석 결과")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "평균 PAE",
                    f"{mean_pae:.2f} Å",
                )

            with col2:
                st.metric(
                    "최대 PAE",
                    f"{max_pae:.2f} Å",
                )

            # -------------------------
            # PAE Heatmap
            # -------------------------

            st.markdown("### 🔥 PAE Heatmap")

            fig, ax = plt.subplots(
                figsize=(8, 7)
            )

            im = ax.imshow(
                pae,
                origin="lower",
                aspect="auto",
            )

            ax.set_xlabel("Residue position")
            ax.set_ylabel("Residue position")
            ax.set_title(
                "Predicted Aligned Error (PAE)"
            )

            fig.colorbar(
                im,
                ax=ax,
                label="PAE (Å)",
            )

            st.pyplot(fig)
            plt.close(fig)

            # -------------------------
            # Residue별 평균 PAE
            # -------------------------

            st.markdown(
                "### 📈 Residue별 평균 PAE"
            )

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            ax.plot(
                pae_df["position"],
                pae_df["mean_PAE"],
            )

            ax.axhline(
                mean_pae,
                linestyle="--",
                label=f"전체 평균 PAE = {mean_pae:.2f} Å",
            )

            ax.set_xlabel("Residue position")
            ax.set_ylabel("Mean PAE (Å)")
            ax.set_title(
                "Mean PAE by Residue"
            )
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

            # -------------------------
            # 종합 주의 영역
            # -------------------------

            st.markdown(
                "## ⚠️ 종합 주의 영역"
            )

            st.info(
                "이 프로젝트에서는 pLDDT가 70 미만이고 "
                "평균 PAE가 전체 평균보다 높은 residue를 "
                "종합 주의 residue로 정의했습니다."
            )

            if attention_positions:
                attention_detail = merged[
                    merged["attention"]
                ].copy()

                st.success(
                    f"총 {len(attention_positions)}개의 "
                    "종합 주의 residue가 확인되었습니다."
                )

                st.dataframe(
                    attention_detail,
                    use_container_width=True,
                )

                st.markdown(
                    "### 주의 영역의 pLDDT"
                )

                fig, ax = plt.subplots(
                    figsize=(12, 5)
                )

                ax.plot(
                    attention_detail["position"],
                    attention_detail["pLDDT"],
                    marker="o",
                    label="pLDDT",
                )

                ax.axhline(
                    70,
                    linestyle="--",
                    label="pLDDT 70",
                )

                ax.set_xlabel(
                    "Residue position"
                )
                ax.set_ylabel("pLDDT")
                ax.set_title(
                    "pLDDT of Attention Region"
                )
                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

            else:
                st.write(
                    "설정한 기준을 동시에 만족하는 "
                    "residue가 없습니다."
                )

            # -------------------------
            # 3D 구조
            # -------------------------

            st.markdown("## 🧬 3D 단백질 구조")

            st.write(
                "pLDDT 기반 신뢰도와 종합 주의 영역을 "
                "서로 다른 방식으로 확인할 수 있습니다."
            )

            structure_text = file_bytes.decode("utf-8")

            tab_plddt, tab_attention = st.tabs(
                ["🎨 pLDDT 신뢰도 3D", "⚠️ 종합 주의 영역 3D"]
            )

            with tab_plddt:
                st.markdown(
                    """
                    <div class="card">
                        <div class="section-title">pLDDT 기반 구조 신뢰도</div>
                        <div class="description">
                            각 residue의 pLDDT 값을 색상으로 표현했습니다.
                            색상이 다를수록 구조 예측 신뢰도 구간이 다릅니다.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
                with legend_col1:
                    st.markdown("🔵 **90–100**  매우 높음")
                with legend_col2:
                    st.markdown("🩵 **70–90**  높음")
                with legend_col3:
                    st.markdown("🟡 **50–70**  낮음")
                with legend_col4:
                    st.markdown("🔴 **0–50**  매우 낮음")

                render_plddt_3d(
                    structure_text,
                    file_name,
                    df,
                )

            with tab_attention:
                st.markdown(
                    """
                    <div class="card">
                        <div class="section-title">pLDDT + PAE 종합 주의 영역</div>
                        <div class="description">
                            이 프로젝트에서 설정한 기준을 만족하는 residue를
                            빨간색으로 강조했습니다.
                            <br>
                            <b>pLDDT &lt; 70 AND 평균 PAE &gt; 전체 평균 PAE</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                attention_rows = merged[merged["attention"]].copy()

                if attention_rows.empty:
                    st.info(
                        "설정한 기준을 동시에 만족하는 residue가 없습니다."
                    )
                else:
                    st.write(
                        f"총 {len(attention_rows)}개의 "
                        "종합 주의 residue가 빨간색으로 표시됩니다."
                    )

                render_attention_3d(
                    structure_text,
                    file_name,
                    attention_rows,
                )

            # -------------------------
            # 최저 pLDDT residue
            # -------------------------

            st.markdown(
                "### ⚠️ 가장 낮은 pLDDT residue"
            )

            st.dataframe(
                df.nsmallest(
                    10,
                    "pLDDT",
                ).reset_index(drop=True),
                use_container_width=True,
            )

            # -------------------------
            # 자동 해석
            # -------------------------

            st.markdown("## 📝 분석 요약")

            if mean_plddt >= 90:
                plddt_summary = (
                    "전체적으로 매우 높은 구조 예측 신뢰도를 "
                    "보입니다."
                )
            elif mean_plddt >= 70:
                plddt_summary = (
                    "전체적으로 양호한 구조 예측 신뢰도를 "
                    "보입니다."
                )
            else:
                plddt_summary = (
                    "전체적인 예측 신뢰도를 해석할 때 "
                    "주의가 필요합니다."
                )

            st.write(
                f"평균 pLDDT는 {mean_plddt:.2f}이며, "
                f"{plddt_summary}"
            )

            if attention_positions:
                st.write(
                    f"pLDDT와 평균 PAE를 종합했을 때 "
                    f"{len(attention_positions)}개의 residue가 "
                    "상대적으로 높은 불확실성을 보이는 "
                    "주의 영역으로 탐지되었습니다."
                )
            else:
                st.write(
                    "설정한 기준을 동시에 만족하는 "
                    "종합 주의 영역은 탐지되지 않았습니다."
                )

            st.caption(
                "※ 주의 영역은 이 프로젝트에서 설정한 "
                "보조 분석 기준에 따른 결과이며, "
                "구조가 실제로 잘못되었다는 의미는 아닙니다."
            )

        except Exception as e:
            st.error(
                "분석 중 오류가 발생했습니다."
            )
            st.exception(e)


# =========================================================
# 결과 예시
# =========================================================

elif page == "📊 결과 예시":

    st.markdown("## 📊 결과 예시")

    st.markdown(
        """
        <div class="card">
        <div class="section-title">예상되는 분석 결과</div>
        <div class="description">
        실제 AlphaFold 구조 파일과 PAE JSON을 업로드하면
        residue별 pLDDT, PAE heatmap, 평균 PAE,
        종합 주의 영역 및 3D 구조를 확인할 수 있습니다.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "이 페이지의 설명은 실제 측정값이 아니라 "
        "Protein Insight의 분석 항목을 설명하기 위한 예시입니다."
    )


# =========================================================
# 비교 분석
# =========================================================

elif page == "⚖️ 비교 분석":

    st.markdown("## ⚖️ 비교 분석")

    st.markdown(
        """
        <div class="card">
        <div class="section-title">
        여러 단백질의 구조 예측 신뢰도를 비교하는 기능
        </div>
        <div class="description">
        현재 버전에서는 개별 단백질 분석을 중심으로 구현되어 있습니다.
        추후 여러 단백질의 평균 pLDDT, 평균 PAE,
        주의 영역 비율 등을 비교할 수 있도록 확장할 수 있습니다.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 자료실
# =========================================================

elif page == "📁 자료실":

    st.markdown("## 📁 자료실")

    with st.expander("pLDDT란?"):
        st.write(
            """
            pLDDT(predicted Local Distance Difference Test)는
            residue별 단백질 구조 예측의 국소적 신뢰도를 나타내는
            지표입니다. 일반적으로 0~100 범위로 표현되며
            값이 높을수록 해당 부분의 예측에 대한 신뢰도가 높다고
            해석합니다.
            """
        )

    with st.expander("PAE란?"):
        st.write(
            """
            PAE(Predicted Aligned Error)는 두 residue 또는
            구조 영역 사이의 상대적인 위치에 대한
            예측 불확실성을 나타내는 지표입니다.
            값이 낮을수록 상대적인 위치 관계에 대한
            예측이 더 확실하다고 해석할 수 있습니다.
            """
        )

    with st.expander("왜 pLDDT와 PAE를 함께 볼까?"):
        st.write(
            """
            pLDDT는 단백질의 각 부분을 얼마나 자신 있게 예측했는지를
            보여주고, PAE는 두 부분의 위치 관계를 얼마나 정확하게
            예측했는지를 보여줍니다.
            따라서 두 지표를 함께 살펴보면 단백질 구조를
            한 가지 기준이 아닌 서로 다른 관점에서 확인할 수 있습니다.
            """
        )

    with st.expander("PDB 파일이란?"):
        st.write(
            """
            PDB(Protein Data Bank) 형식은 단백질과 같은 생체분자의
            3차원 구조 정보를 저장하는 파일 형식입니다.
            원자의 위치와 residue, chain 등의 구조 정보를 포함하며,
            단백질의 3차원 구조를 확인하거나 분석할 때 사용됩니다.
            """
        )

    with st.expander("mmCIF 파일이란?"):
        st.write(
            """
            mmCIF(PDBx/mmCIF)는 단백질 구조 정보를 저장하기 위한
            형식으로, 현재 PDB 구조 데이터의 표준 형식입니다.
            정보를 항목과 표 형태로 체계적으로 저장할 수 있어
            큰 구조나 복잡한 구조 데이터도 표현할 수 있습니다.
            """
        )

    st.caption(
        "※ PDB와 mmCIF는 단백질 구조 정보를 저장하는 파일 형식이며, "
        "Protein Insight는 이 파일에서 구조 정보와 pLDDT 데이터를 읽어 분석합니다."
    )


# =========================================================
# 소개
# =========================================================

elif page == "ℹ️ 소개":

    st.markdown("## ℹ️ Protein Insight 소개")

    st.markdown(
        """
        <div class="card">
        <div class="section-title">연구 과정</div>
        <div class="description">
        <b>1. 탐구 동기</b><br>
        단백질 구조 예측 결과를 단순히 확인하는 것을 넘어
        예측 결과 자체의 신뢰도를 분석하고자 했습니다.
        <br><br>

        <b>2. 분석 지표 탐색</b><br>
        AlphaFold의 pLDDT와 PAE를 조사하고
        각각이 어떤 정보를 제공하는지 분석했습니다.
        <br><br>

        <b>3. 분석 방법 설계</b><br>
        residue별 pLDDT와 평균 PAE를 분석하고,
        두 지표를 결합한 종합 주의 영역 기준을 설계했습니다.
        <br><br>

        <b>4. Python 구현</b><br>
        Bio.PDB, NumPy, Pandas, Matplotlib 등을 이용하여
        구조 파일과 PAE 데이터를 분석했습니다.
        <br><br>

        <b>5. 시각화</b><br>
        pLDDT 그래프, PAE heatmap,
        pLDDT 색상 기반 3D 구조와
        종합 주의 영역 3D 구조를 구현했습니다.
        <br><br>

        <b>6. 웹사이트 구현</b><br>
        Streamlit을 이용하여 분석 과정을
        하나의 웹 기반 도구로 통합했습니다.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
        <div class="section-title">핵심 아이디어</div>
        <div class="description">
        AlphaFold 구조 예측<br>
        ↓<br>
        pLDDT / PAE 데이터 분석<br>
        ↓<br>
        예측 불확실성이 높은 영역 탐색<br>
        ↓<br>
        그래프 및 3D 구조 시각화
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 도움말
# =========================================================

elif page == "❓ 도움말":

    st.markdown("## ❓ 도움말")

    with st.expander("① 어떤 파일을 준비해야 하나요?"):
        st.write(
            """
            AlphaFold에서 제공되는 단백질 구조 파일(PDB 또는 mmCIF)
            과 대응되는 PAE JSON 파일이 필요합니다.
            """
        )

    with st.expander("② 종합 주의 영역은 무엇인가요?"):
        st.write(
            """
            이 프로젝트에서는 pLDDT가 70 미만이고
            해당 residue의 평균 PAE가 전체 평균 PAE보다 높은 경우를
            종합 주의 residue로 정의합니다.
            이는 프로젝트에서 설정한 보조 분석 기준이며,
            실제 구조가 틀렸다는 뜻은 아닙니다.
            """
        )

    with st.expander("③ 결과를 어떻게 해석하나요?"):
        st.write(
            """
            pLDDT는 개별 residue의 국소적인 구조 예측 신뢰도를,
            PAE는 residue 또는 영역 사이의 상대적인 위치
            예측 불확실성을 보여줍니다.
            두 지표를 함께 보는 것이 중요합니다.
            """
        )


# =========================================================
# Footer
# =========================================================

st.divider()

st.markdown(
    "<div style='text-align:center; color:#718096; font-size:13px;'>"
    "From Experiment to Insight　|　"
    "AlphaFold 구조 예측 결과를 보다 정확하게 해석하기 위한 도구"
    "</div>",
    unsafe_allow_html=True,
)
