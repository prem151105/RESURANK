from setuptools import setup, find_packages

setup(
    name="agentnet",
    version="0.1.0",
    description="Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems",
    author="AgentNet Team",
    author_email="info@agentnet.ai",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)