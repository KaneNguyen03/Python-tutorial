import streamlit as st

st.title("MY PROJECT")
st.header("This is a header")
st.subheader("This is a sub header")
st.text("This is a text")
st.markdown("This is a markdown")
st.latex(r''' e^{i\pi} + 1 = 0 ''')
st.code('''def fact(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n*fact(n-1)''')
st.write("This is a write")
st.success("This is a success message")
st.info("This is an info message")
st.warning("This is a warning message")
st.error("This is an error message")

st.exception("This is an exception message")            


st.badge("New")
st.badge("Success", icon=":material/check:", color="green")
st.badge("Warning", icon=":material/warning:", color="yellow")

st.markdown(":violet-badge[:material/star: Favorite] \ :orange-badge[Need review] \ :gray-badge[Deprecated]")