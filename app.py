from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# تحميل النموذج
model = joblib.load('xgboost_model.pkl')

# تحميل البيانات لاستخراج الفئات الفريدة وتطبيق Label Encoding
data = pd.read_csv('Egypt_Houses_Price.csv')  # تأكد من تحديث اسم الملف إذا كان مختلفًا
data = data.replace({
    'Stand Alone Villa': 'Standalone Villa',
    'Twin house': 'Twin House'
})
# إعداد القيم الفريدة للعناصر الفئوية
cities = data['City'].unique().tolist()
compounds = data['Compound'].unique().tolist()
types = data[data['Type'] != 'Unknown']['Type'].unique().tolist()
levels = data[data['Level'] != 'Unknown']['Level'].unique().tolist()

# إنشاء Label Encoders لكل عمود فئوي
encoders = {}
for col in ['Type', 'Level', 'Compound', 'City']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# الأعمدة المتوقعة بعد التشفير
expected_columns = ['Type', 'Bedrooms', 'Bathrooms', 'Area', 'Level', 'Compound', 'City']

@app.route('/')
def home():
    # تمرير القيم الفريدة إلى القالب لعرضها في القائمة المنسدلة
    return render_template('index.html', cities=cities, compounds=compounds, types=types, levels=levels)

@app.route('/predict', methods=['POST'])
def predict():
    # الحصول على البيانات من النموذج
    data = request.form.to_dict()

    # تحويل البيانات إلى DataFrame
    input_data = pd.DataFrame([data])

    # تحويل الأعمدة العددية
    input_data['Bedrooms'] = int(input_data['Bedrooms'])
    input_data['Bathrooms'] = int(input_data['Bathrooms'])
    input_data['Area'] = float(input_data['Area'])

    # تحويل الأعمدة الفئوية باستخدام Label Encoding
    for col in ['Type', 'Level', 'Compound', 'City']:
        input_data[col] = encoders[col].transform([input_data[col]])[0]

    # التأكد من تطابق الأعمدة مع الأعمدة المتوقعة
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    # التنبؤ باستخدام النموذج
    prediction = model.predict(input_data)[0]

    # إرسال النتيجة إلى القالب لعرضها
    return render_template('index.html', cities=cities, compounds=compounds, types=types, levels=levels, prediction=prediction , data=request.form)

if __name__ == '__main__':
    app.run(debug=True)
