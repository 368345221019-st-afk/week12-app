from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ---------------------------------------------------------------
st.set_page_config(page_title="ทำนายการรอดชีวิต Titanic", page_icon="🚢")

MODEL_PATH = Path(__file__).parent / "titanic_tree.joblib"

# ---------------------------------------------------------------
# ค่าที่ใช้แปลงข้อมูลให้เหมือนตอนเทรนใน Colab
# ถ้าผลทำนายดูแปลก ๆ ให้ตรวจ/แก้ค่าในส่วนนี้ให้ตรงกับโค้ดเตรียมข้อมูลของคุณ
# ---------------------------------------------------------------
PCLASS_OFFSET = 1          # โมเดลเห็น Pclass เป็น 0,1,2  (= ชั้นโดยสาร - 1)
AGE_MIN, AGE_MAX = 0.42, 80.0        # ค่า min/max ของ Age ในชุดเทรน (Min-Max scaling)
FARE_MIN, FARE_MAX = 0.0, 512.3292   # ค่า min/max ของ Fare ในชุดเทรน (Min-Max scaling)


@st.cache_resource
def load_model(path: Path):
    """โหลดโมเดลครั้งเดียว แล้วเก็บไว้ใช้ซ้ำ"""
    return joblib.load(path)


def min_max(value: float, lo: float, hi: float) -> float:
    """แปลงค่าให้อยู่ช่วง 0-1 ตามสูตร (x - min) / (max - min)"""
    return min(max((value - lo) / (hi - lo), 0.0), 1.0)


def build_features(pclass, sex, age, fare, family_size, feature_order):
    """สร้าง DataFrame 1 แถว ให้ชื่อคอลัมน์และลำดับตรงกับตอนเทรน"""
    row = {
        "Pclass": pclass - PCLASS_OFFSET,
        "Sex_female": 1 if sex == "หญิง" else 0,
        "Age": min_max(age, AGE_MIN, AGE_MAX),
        "Fare": min_max(fare, FARE_MIN, FARE_MAX),
        "FamilySize": family_size,
    }
    return pd.DataFrame([row])[list(feature_order)]


# ---------------------------------------------------------------
# โหลดโมเดล
# ---------------------------------------------------------------
st.title("🚢 ทำนายการรอดชีวิตของผู้โดยสาร Titanic")
st.write("กรอกข้อมูลผู้โดยสาร แล้วกดปุ่มเพื่อให้โมเดล Decision Tree ทำนายผล")

if not MODEL_PATH.exists():
    st.error(
        f"ไม่พบไฟล์ `{MODEL_PATH.name}` ในโฟลเดอร์เดียวกับ app.py\n\n"
        f"ให้วางไฟล์ไว้ที่: `{MODEL_PATH.parent}`"
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("โหลดโมเดลไม่สำเร็จ (มักเกิดจากเวอร์ชัน scikit-learn ไม่ตรงกับตอนเทรน)")
    st.exception(e)
    st.stop()

# ---------------------------------------------------------------
# ฟอร์มกรอกข้อมูล
# ---------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "ชั้นโดยสาร (Pclass)",
        options=[1, 2, 3],
        index=2,
        format_func=lambda x: f"ชั้น {x}",
    )
    sex = st.radio("เพศ", options=["ชาย", "หญิง"], horizontal=True)
    age = st.slider("อายุ (ปี)", min_value=0.5, max_value=80.0, value=30.0, step=0.5)

with col2:
    fare = st.number_input(
        "ค่าโดยสาร (Fare)", min_value=0.0, max_value=520.0, value=32.0, step=1.0
    )
    family_size = st.number_input(
        "จำนวนสมาชิกครอบครัวที่เดินทางด้วยกัน (รวมตัวเอง)",
        min_value=1,
        max_value=11,
        value=1,
        step=1,
    )

# ---------------------------------------------------------------
# ทำนายผล
# ---------------------------------------------------------------
if st.button("ทำนายผล", type="primary"):
    X = build_features(pclass, sex, age, fare, family_size, model.feature_names_in_)

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    p_survive = float(proba[list(model.classes_).index(1)])

    if pred == 1:
        st.success(f"ผลทำนาย: **รอดชีวิต** 🎉  (ความน่าจะเป็น {p_survive:.1%})")
    else:
        st.error(f"ผลทำนาย: **ไม่รอดชีวิต** 😢  (ความน่าจะเป็นที่จะรอด {p_survive:.1%})")

    st.progress(p_survive, text=f"โอกาสรอดชีวิต {p_survive:.1%}")

    with st.expander("ดูข้อมูลที่ส่งเข้าโมเดลจริง (สำหรับตรวจสอบ)"):
        st.dataframe(X)
        st.caption(
            "เทียบกับ X_train.head() ใน Colab ได้ ถ้าสเกลของ Pclass/Age/Fare "
            "ไม่ตรงกัน ให้แก้ค่าคงที่ที่ส่วนบนของไฟล์ app.py"
        )
