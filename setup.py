from setuptools import find_packages, setup


def read_requirements(filename: str):
    with open(filename) as requirements_file:
        import re

        pattern = (
            r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git"
        )

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL deps should be handled."""
            m = re.match(pattern, req)
            if m is None:
                return req
            else:
                return f"{m.group('name')} @ {req}"

        requirements = []
        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            requirements.append(fix_url_dependencies(line))
    return requirements


setup(
    name="uhd-eeg",
    version="0.1",
    author="ARAYA",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    package_data={"uhd_eeg": ["py.typed"]},
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.8",
)
