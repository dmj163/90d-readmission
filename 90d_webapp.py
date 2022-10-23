import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import sklearn
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

# 页面内容设置
# 页面名称
st.set_page_config(page_title="90d readmission", layout="wide")
# 标题
st.title('An online web-app for predicting 90-day readmission')

st.markdown('_This is a webApp to predict the risk of 90-day unplanned all-cause readmission\
         based on several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction._',)
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
        return "Single peak E"
    if x == 1:
        return "0-1"
    if x == 2:
        return '1-2'
    if x == 3:
        return '>2'


@st.cache
def predict_probability(model, df):
    y_pred = model.predict_proba(df)
    return y_pred[:, 1]


# 导入模型
model = joblib.load('save/cb_90d_14_isotonic.pkl')
model1 = joblib.load('save/cb_90d_14.pkl')
st.sidebar.title("Features")

# 设置各项特征的输入范围和选项
LOS = st.sidebar.slider(label='Length_of_stay', min_value=0,
                                  max_value=100,
                                  value=20,
                                  step=1)


NT_proBNP = st.sidebar.number_input(label='NT-proBNP', min_value=0.0000,
                       max_value=30000.0000,
                       value=100.0000,
                       step=1.0000)

INR = st.sidebar.number_input(label='INR', min_value=0.00,
                       max_value=20.00,
                       value=10.00,
                       step=0.01)

NEUT = st.sidebar.slider(label='NEUT%', min_value=0.00,
                                   max_value=1.00,
                                   value=0.05,
                                   step=0.01)

TT= st.sidebar.number_input(label='TT', min_value=0.00,
                              max_value=100.00,
                              value=10.0,
                              step=1.00)

BUN = st.sidebar.number_input(label='BUN', min_value=0.00,
                            max_value=100.00,
                            value=0.00,
                            step=0.01)

RBC = st.sidebar.number_input(label='RBC', min_value=0.00,
                            max_value=10.00,
                            value=0.01,
                            step=0.01)

HCI = st.sidebar.number_input(label='HCI', min_value=0.0,
                            max_value=100.0,
                            value=30.0,
                            step=0.1)

Scr = st.sidebar.number_input(label='Scr', min_value=0.0,
                            max_value=1000.0,
                            value=0.0,
                            step=0.001)

CK_MB = st.sidebar.number_input(label='CK-MB', min_value=0.0,
                            max_value=2000.0,
                            value=0.0,
                            step=0.1)

ALT = st.sidebar.number_input(label='ALT', min_value=0.000,
                            max_value=1000.000,
                            value=1.000,
                            step=0.100)

FS = st.sidebar.number_input(label='FS', min_value=0,
                            max_value=100,
                            value=30,
                            step=1,)

E_A = st.sidebar.selectbox(label='E/A', options=[0, 1, 2, 3], format_func=lambda x: option_name1(x), index=0)

Beta_blockers = st.sidebar.selectbox(label='Beta-blockers', options=[0, 1], format_func=lambda x: option_name(x), index=0)



features = {'LOS': LOS,
            'BNP': NT_proBNP,
            'INR': INR,
            'NEUT%': NEUT,
            'TT': TT,
            'BUN': BUN,
            'RBC': RBC,
            'HCI':HCI,
            'Scr': Scr,
            'CK-MB': CK_MB,
            'ALT': ALT,
            'FS':FS,
            'E/A':E_A,
            'Beta-blockers':Beta_blockers,
}

features_df = pd.DataFrame([features])
# features_df=features_df.round(4)
# print(features_df.round(4))
#显示输入的特征
st.table(features_df)

#显示预测结果与shap解释图
if st.button('Predict'):
    prediction = predict_probability(model1, features_df)
    st.write("the probability of readmission:")
    st.success(round(prediction[0], 3))
    explainer = shap.TreeExplainer(model1)
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


