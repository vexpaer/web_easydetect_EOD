from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

app = Flask(__name__)

# === 配置 ===
DIAG_MODEL_PATH = "诊断Catboost.pkl"
PROG_MODEL_PATH = "预后RF.pkl"
DIAG_SHAP_PATH = "shap诊断.pkl"
PROG_SHAP_PATH = "shap预后.pkl"

# 结果图片路径
RESULT_IMAGES = {
    "diagnosis_high": "诊断患病.png",
    "diagnosis_low": "诊断健康.png",
    "prognosis_poor": "预后差.png",
    "prognosis_good": "预后好.png"
}

DIAG_THRESHOLD = 0.53
PROG_THRESHOLD = -12.03

# 在应用启动时加载模型和解释器
diag_model = joblib.load(DIAG_MODEL_PATH)
prog_model = joblib.load(PROG_MODEL_PATH)
diag_explainer = joblib.load(DIAG_SHAP_PATH)
prog_explainer = joblib.load(PROG_SHAP_PATH)

# 特征顺序（与模型训练时一致）
FEATURE_NAMES = [
    'age', 'educl', 'gender', 'hibpe', 'stroke', 'diabe', 'cancre', 
    'lunge', 'hearte', 'arthre', 'psyche', 'height', 'weight', 
    'smokev', 'smoken', 'drinkev', 'bmi', 'family_size', 
    'work', 'hearaid', 'rural', 'lgrip', 'rgrip', 'vgactx', 'pain', 'marry'
]

def encode_image_to_base64(image_path):
    """将图片编码为base64字符串"""
    if not os.path.exists(image_path):
        print(f"警告: 图片文件不存在: {image_path}")
        return None
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"读取图片失败: {str(e)}")
        return None

# 预加载结果图片
result_images_base64 = {}
for key, path in RESULT_IMAGES.items():
    result_images_base64[key] = encode_image_to_base64(path)

def generate_shap_plot(explainer, X, target_score, feature_names):
    """
    生成SHAP力导向图并返回base64编码的图像
    """
    # 计算原始SHAP值
    shap_values_original = explainer(X)
    
    # 获取原始SHAP值和基线值
    original_shap_values = shap_values_original[0].values
    original_base_value = shap_values_original[0].base_values
    
    # 计算调整比例
    if np.sum(np.abs(original_shap_values)) > 0:
        adjustment_factor = (target_score - original_base_value) / np.sum(original_shap_values)
        adjusted_shap_values = original_shap_values * adjustment_factor
    else:
        adjusted_shap_values = original_shap_values
        original_base_value = target_score
    
    # 创建自定义的SHAP Explanation对象
    custom_explanation = shap.Explanation(
        values=adjusted_shap_values,
        base_values=original_base_value,
        data=shap_values_original[0].data,
        feature_names=feature_names
    )
    
    # 绘制力图（移除标题）
    plt.figure(figsize=(12, 6))
    shap.force_plot(
        custom_explanation,
        matplotlib=True,
        text_rotation=45,
        show=False
    )
    # 移除图中的f(x)文本
    ax = plt.gca()
    for text in ax.texts:
        if 'f(x)' in text.get_text():
            text.set_visible(False)
    
    plt.tight_layout()
    
    # 将图像转换为base64字符串
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# === 处理页面请求 ===
@app.route('/', methods=['GET'])
def index():
    """中文版页面（首页）"""
    return render_template('index.html')

@app.route('/index.html', methods=['GET'])
def index_cn():
    """中文版页面（直接访问）"""
    return render_template('index.html')

@app.route('/index2.html', methods=['GET'])
def index_en():
    """英文版页面"""
    return render_template('index2.html')

# === 处理预测请求 ===
@app.route('/predict', methods=['POST'])
def predict():
    # 获取模型类型和前端传来的数据
    model_type = request.form.get('model_type', 'diagnosis')  # 默认为诊断
    input_data = request.form.get('data').split(',')
    
    # 确保数据长度正确
    if len(input_data) != len(FEATURE_NAMES):
        return jsonify({
            "error": f"需要 {len(FEATURE_NAMES)} 个特征值，但收到 {len(input_data)} 个"
        })
    
    try:
        # 转换为浮点数，处理空值
        processed_data = []
        for i, value in enumerate(input_data):
            if value.strip() == "":
                processed_data.append(np.nan)
            else:
                processed_data.append(float(value))
        
        # 创建DataFrame
        X = pd.DataFrame([processed_data], columns=FEATURE_NAMES)
        
        if model_type == 'diagnosis':
            # === 诊断模型处理 ===
            try:
                diag_feats = list(diag_model.feature_names_in_)
                X_diag = X.reindex(columns=diag_feats).fillna(X.median())
            except AttributeError:
                diag_feats = getattr(diag_model, "feature_names_", FEATURE_NAMES)
                X_diag = X.reindex(columns=diag_feats).fillna(X.median())
            
            # 诊断得分
            if hasattr(diag_model, "predict_proba"):
                diag_score = diag_model.predict_proba(X_diag)[0, 1]
            else:
                diag_score = diag_model.predict(X_diag)[0]
            
            # 根据阈值确定分类
            diag_label = "高" if diag_score > DIAG_THRESHOLD else "低"
            
            # 生成SHAP图
            shap_image = generate_shap_plot(diag_explainer, X_diag, diag_score, diag_feats)
            
            # 获取结果图片
            result_key = f"diagnosis_{'high' if diag_label == '高' else 'low'}"
            result_image = result_images_base64.get(result_key)
            
            # 返回结果
            return jsonify({
                "model_type": "diagnosis",
                "diagnosis_label": diag_label,
                "diagnosis_score": f"{diag_score:.4f}",
                "shap_image": shap_image,
                "result_image": result_image
            })
            
        elif model_type == 'prognosis':
            # === 预后模型处理 ===
            try:
                prog_feats = list(prog_model.feature_names_in_)
                X_prog = X.reindex(columns=prog_feats).fillna(X.median())
            except AttributeError:
                prog_feats = getattr(prog_model, "feature_names_", FEATURE_NAMES)
                X_prog = X.reindex(columns=prog_feats).fillna(X.median())
            
            # 预后风险分
            raw_score = prog_model.predict(X_prog)[0]
            prog_score = -raw_score  # 取负值
            
            # 根据阈值确定分类
            prog_label = "较差" if prog_score > PROG_THRESHOLD else "较好"
            
            # 生成SHAP图
            shap_image = generate_shap_plot(prog_explainer, X_prog, raw_score, prog_feats)
            
            # 获取结果图片
            result_key = f"prognosis_{'poor' if prog_label == '较差' else 'good'}"
            result_image = result_images_base64.get(result_key)
            
            # 返回结果
            return jsonify({
                "model_type": "prognosis",
                "prognosis_label": prog_label,
                "prognosis_score": f"{prog_score:.4f}",
                "shap_image": shap_image,
                "result_image": result_image
            })
        
    except Exception as e:
        return jsonify({
            "error": f"处理数据时出错: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)