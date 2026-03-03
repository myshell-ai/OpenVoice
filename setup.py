from setuptools import setup, find_packages
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

def get_requirements(path):
    """Load and clean requirements file."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

# Paths
readme_path = os.path.join(base_dir, "README.md")
requirements_path = os.path.join(base_dir, "requirements.txt")

# Load README
with open(readme_path, "r", encoding="utf-8") as f:
    long_description = f.read().strip()

setup(
    name='OpenVoice',
    version='0.1.0',
    description='Instant voice cloning by MyShell - Enhanced by Gedeon.',
    long_description=long_description,
    long_description_content_type='text/markdown',

    keywords=[
        'text-to-speech',
        'tts',
        'voice-clone',
        'zero-shot-tts'
    ],

    url='https://github.com/beethogedeon/OpenVoice',
    project_urls={
        'Documentation': 'https://github.com/beethogedeon/OpenVoice/blob/main/docs/USAGE.md',
        'Changes': 'https://github.com/beethogedeon/OpenVoice/releases',
        'Code': 'https://github.com/beethogedeon/OpenVoice',
        'Issue tracker': 'https://github.com/beethogedeon/OpenVoice/issues',
    },

    author='MyShell, Gedeon GBEDONOU',
    author_email='ethan@myshell.ai, beethovengedeon@gmail.com',

    license='MIT License',
    packages=find_packages(),
    python_requires='>=3.9',

    install_requires=get_requirements(requirements_path),
)
