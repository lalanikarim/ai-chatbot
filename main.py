import streamlit as st
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def main():
    st.set_page_config(
        page_title="Your own Chat!"
    )
    st.header("Your own Chat!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
            model_path="mistral-7b-instruct-v0.1.Q4_0.gguf",
            temperature=0,
            max_tokens=512,
            top_p=1,
            callback_manager=callback_manager,
            verbose=True,
            )

    template = """
    You are a funny AI bot who answers questions in a couple of lines.

    {question}
    """

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    if prompt := st.chat_input("Your message here", key="user_input"):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        response = llm_chain.run(prompt)
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == '__main__':
    main()
