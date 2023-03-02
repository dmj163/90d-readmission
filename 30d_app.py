import streamlit as st
import pandas as pd
import joblib
# import shap
import matplotlib.pyplot as plt
import sklearn

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 页面内容设置
# 页面名称
st.set_page_config(page_title="30d readmission", layout="wide")
# 标题
st.title('An online web-app for predicting 30-day readmission')

st.markdown('_This is a webApp to predict the risk of 30-day unplanned all-cause readmission\
         based on several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction._', )
st.markdown('## *Input Data:*')
# 隐藏底部水印
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            <![]()yle>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)


def option_name(x):
    if x == 0:
        return "no"
    if x == 1:
        return "yes"


def option_name1(x):
    if x == 0:
        return "UEBMI"
    if x == 1:
        return "NRCMS"
    if x == 2:
        return 'Commercial Health Insurance'
    if x == 3:
        return 'Self-paid treatment'
    if x == 4:
        return 'Others'


@st.cache
def predict_probability(model, df):
    y_pred = model.predict_proba(df)
    return y_pred[:, 1]


# 导入模型
model = joblib.load('save/cb_150220.pkl')
st.sidebar.title("Features")

# 设置各项特征的输入范围和选项
age = st.sidebar.slider(label='age', min_value=0.0,
                        max_value=18.0,
                        value=5.0,
                        step=0.5)

Insurancetype = st.sidebar.selectbox(label='Insurance type', options=[0, 1, 2, 3, 4],
                                     format_func=lambda x: option_name1(x), index=0)

ACEIARB = st.sidebar.selectbox(label='ACEI/ARB', options=[0, 1], format_func=lambda x: option_name(x), index=0)
cardiacsurgery = st.sidebar.selectbox(label='cardiac surgery', options=[0, 1], format_func=lambda x: option_name(x),
                                      index=0)

LOS = st.sidebar.slider(label='Length_of_stay', min_value=0,
                        max_value=100,
                        value=20,
                        step=1)

BNP = st.sidebar.number_input(label='BNP', min_value=0.0000,
                              max_value=30000.0000,
                              value=100.0000,
                              step=1.0000)

IRT = st.sidebar.number_input(label='IRT', min_value=0.00,
                              max_value=300.00,
                              value=50.00,
                              step=0.01)

LYMPH = st.sidebar.slider(label='LYMPH%', min_value=0.00,
                          max_value=1.00,
                          value=0.05,
                          step=0.01)


UA = st.sidebar.number_input(label='UA', min_value=0.00,
                             max_value=2000.00,
                             value=0.00,
                             step=0.01)

PLT = st.sidebar.number_input(label='PLT', min_value=0,
                              max_value=1000,
                              value=200,
                              step=0)

GGT = st.sidebar.number_input(label='GGT', min_value=0.0,
                              max_value=1000.0,
                              value=30.0,
                              step=0.1)

Scr = st.sidebar.number_input(label='Scr', min_value=0.0,
                              max_value=1000.0,
                              value=0.0,
                              step=0.1)

CK_MB = st.sidebar.number_input(label='CK-MB', min_value=0.0,
                                max_value=500.0,
                                value=0.0,
                                step=0.1)

ALT = st.sidebar.number_input(label='ALT', min_value=0.0,
                              max_value=5000.0,
                              value=100.00,
                              step=0.01)

ALB = st.sidebar.number_input(label='ALB', min_value=0,
                              max_value=100,
                              value=30,
                              step=1)



features = {'ALT': ALT,
            'PLT': PLT,
            'CK-MB': CK_MB,
            'cardiac surgery': cardiacsurgery,
            'BNP': BNP,
            'ALB': ALB,
            'LYMPH%': LYMPH,
            'GGT': GGT,
            'Scr': Scr,
            'Insurance type': Insurancetype,
            'UA': UA,
            'age': age,
            'IRT': IRT,
            'ACEI/ARB': ACEIARB,
            'LOS': LOS,
            }

features_df = pd.DataFrame([features])
# features_df=features_df.round(4)
# print(features_df.round(4))
# 显示输入的特征
st.table(features_df)

# 显示预测结果与shap解释图
if st.button('Predict'):
    prediction = predict_probability(model, features_df)
    st.write("the risk of readmission:")
    st.success(round(prediction[0], 3))
'''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)
    shap.force_plot(explainer.expected_value, shap_values[0], features_df, matplotlib=True, show=False)
    plt.subplots_adjust(top=0.67,
                        bottom=0.0,
                        left=0.1,
                        right=0.9,
                        hspace=0.2,
                        wspace=0.2)
    plt.savefig('test_shap.png')

    st.image('test_shap.png', caption='Individual prediction explanation', use_column_width=True)
     '''
