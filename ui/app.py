import streamlit as st
import requests

st.set_page_config(page_title="Rice Disease Assistant", page_icon="🌾")

st.title("🌾 Rice Disease Assistant")
st.write("Hệ thống hỏi đáp bệnh hại lúa (LLM + RAG)")

question = st.text_input(
    "Nhập triệu chứng bệnh lúa:",
    placeholder="Ví dụ: lúa bị vàng lá, sinh trưởng kém..."
)

if question:
    with st.spinner("Đang phân tích tài liệu..."):
        res = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"question": question},
            timeout=120
        )

    if res.status_code == 200:
        st.markdown("### 📌 Kết quả")
        st.write(res.json()["answer"])
    else:
        st.error("Lỗi khi gọi API")
