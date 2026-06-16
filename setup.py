from setuptools import setup, find_packages

setup(
    name="banking-transaction-prediction",
    version="1.0.0",
    author="Senior Data Scientist",
    author_email="datascientist@banking.com",
    description="Production-grade ML solution for banking transaction prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/banking-transaction-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
        "shap>=0.42.0",
        "flask>=2.3.0",
        "streamlit>=1.27.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8", "pylint"],
        "mlops": ["mlflow", "prometheus-client", "grafana-api"],
    },
)
