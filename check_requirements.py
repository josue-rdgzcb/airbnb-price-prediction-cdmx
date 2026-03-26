# check_requirements.py
# Script to verify installation and versions of key libraries

import importlib

# List of libraries to check
libraries = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "joblib",
    "geopandas",
    "scipy"
]

print("🔍 Checking installed libraries...\n")

for lib in libraries:
    try:
        module = importlib.import_module(lib)
        version = getattr(module, "__version__", "Version not found")
        print(f"{lib}: {version}")
    except ImportError:
        print(f"{lib}: ❌ Not installed")

print("\n✅ Check complete.")

