from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="ccp-in-pytorch",
        author='Xavier Mignot, Pamela Toman',
        author_email='xmignot@paloaltonetworks.com, ptoman@paloaltonetworks.com',
        version="0.1",
        description="Pytorch implementation of CCP.",
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            "torch==2.0.0",
            "torchvision==0.15.1",
            "numpy==1.21.6",
            "pandas==1.3.5",
        ],
        extras_require={
            "tests": ["pytest"],
            "typing": ["types-setuptools", "types-tqdm"],
            "examples": [
                "bpemb",
                "torchtext==0.15.1",
                "torchdata==0.6.0",  # pegged because of frequent API changes (in beta)
                "tqdm",
                "jupyter",
            ],
        },
        python_requires=">=3, <4",
        package_data={"pytorch_ccp": ["ccp/py.typed"]},
    )
