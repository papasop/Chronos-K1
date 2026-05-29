from setuptools import find_packages, setup


setup(
    name="k1-manifold-core",
    version="0.4.0",
    description="Numerical core for the K=1 chronogeometrodynamics framework.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=["numpy>=1.24"],
    extras_require={"dev": ["pytest>=7.0", "sympy>=1.12", "matplotlib>=3.7"]},
)
