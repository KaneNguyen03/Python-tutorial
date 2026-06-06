import streamlit as st
from fact import fact

def main():
    st.title('Factorial Caculator')
    number = st.number_input("Enter a number:",
                             min_value = 0,
                             max_value = 900)
    
    if st.button("Calculate"):
        result = fact(number)
        st.write(f"The factorial of {number} is {result}.")
        st.balloons()

if __name__ == "__main__":
    main()