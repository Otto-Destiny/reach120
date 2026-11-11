import streamlit as st

import appdata
from reach120 import assistant

state = appdata.current_state()
st.title("Ask")
st.caption("Questions about access and upgrades in this state. The assistant can only answer by running Reach120's own "
           "tools, and an answer is shown only if every number in it appears in those tools' results.")

client = assistant.LLMClient.from_settings()
if client is None:
    st.info("The assistant is off because no language-model key is set. Everything else works without it.\n\n"
            "To switch it on for free: create a key at console.groq.com, set it as the environment variable "
            "`REACH120_LLM_API_KEY` (or `GROQ_API_KEY`), and restart the app. Any OpenAI-compatible provider works: "
            "set `REACH120_LLM_BASE_URL` and `REACH120_LLM_MODEL` too.")
    st.stop()
st.caption(f"Model: `{client.model}` at {client.base_url}")

history_key = f"chat_{state}"
history = st.session_state.setdefault(history_key, [])
for turn in history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

examples = ["What share of women are more than two hours from care?",
            "Which five LGAs are worst served?",
            "With ₦3 billion, which hospitals should we upgrade?"]
question = st.chat_input("Ask about access or upgrades")
if not history and not question:
    st.write("Try: " + " · ".join(f"*{q}*" for q in examples))
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Running the tools"):
            prior = [{"role": t["role"], "content": t["content"]} for t in history[-6:]]
            out = assistant.answer(state, question, history=prior, client=client)
        if out.get("error"):
            st.error(out["error"])
            text = None
        else:
            text = out["answer"]
            st.markdown(text)
        if out.get("tool_calls"):
            with st.expander(f"How this was answered: {len(out['tool_calls'])} tool calls"
                             + (f", {out.get('numbers_checked', 0)} numbers checked" if out.get("verified") else "")):
                for c in out["tool_calls"]:
                    st.code(f"{c['tool']}({c['arguments']})", language="python")
    history.append({"role": "user", "content": question})
    if text:
        history.append({"role": "assistant", "content": text})
