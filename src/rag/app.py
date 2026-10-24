"""Small Streamlit interface for the RAG demonstration."""

from __future__ import annotations

import streamlit as st

from rag.config import get_settings
from rag.generator import ask


st.set_page_config(page_title="课程 RAG 问答", page_icon="📚", layout="wide")
st.title("计算机网络课程 RAG 问答")
mode = st.selectbox("问答模式", ("rag_rerank", "rag", "llm_direct"), format_func=lambda value: {
    "rag_rerank": "RAG + Reranker", "rag": "基础 RAG", "llm_direct": "直接模型",
}[value])
question = st.text_area("问题", placeholder="例如：TCP 为什么需要三次握手？")
if st.button("提问", type="primary", disabled=not question.strip()):
    with st.spinner("正在检索与生成…"):
        try:
            result = ask(question, mode=mode, settings=get_settings(), debug=True)
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.subheader("回答")
            st.write(result.answer)
            if result.insufficient_evidence:
                st.warning("当前知识库证据不足。")
            st.caption(f"耗时 {result.latency_ms} ms；证据校验：{'通过' if result.validation['passed'] else '未通过'}")
            if result.validation["issues"]:
                st.error("\n".join(result.validation["issues"]))
            st.subheader("来源")
            for citation in result.citations:
                st.write(f"{citation.source}，第 {citation.page_start}-{citation.page_end} 页（{citation.chunk_id}）")
            with st.expander("检索调试信息"):
                for item in result.retrieved_chunks:
                    st.markdown(f"**{item.chunk.id}** · vector rank {item.vector_rank} · rerank {item.rerank_score}")
                    st.write(item.chunk.text)
