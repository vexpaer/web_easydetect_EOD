easydetect - EOD 网页展示项目

所有模型以及运算代码来源 - lsj

网站搭建 - Vexpaer

使用方式 - 运行app.py或点击点我运行.bat即可,网页端口为5000

项目所需的主要 Python 库：

库名称	版本	用途描述
Flask	>=2.0.0	Web 框架
scikit-learn	>=1.0.0	机器学习工具
pandas	>=1.3.0	数据处理
numpy	>=1.21.0	数值计算
shap	>=0.40.0	模型解释
matplotlib	>=3.4.0	数据可视化
joblib	>=1.0.0	模型加载


文件结构
easydetect-EOD/
├── app.py                  # Flask 应用主文件
├── templates/
│   ├── index.html           # 中文版界面
│   └── index2.html          # 英文版界面
├── 点我运行.bat              # 点我运行
├── 诊断Catboost.pkl         # 诊断模型
├── 预后RF.pkl               # 预后模型
├── shap诊断.pkl             # 诊断模型 SHAP 解释器
├── shap预后.pkl             # 预后模型 SHAP 解释器
├── 诊断患病.png             # 高风险诊断结果图片
├── 诊断健康.png             # 低风险诊断结果图片
├── 预后差.png               # 较差预后结果图片
├── 预后好.png               # 较好预后结果图片
└── README.md                # 项目说明文件