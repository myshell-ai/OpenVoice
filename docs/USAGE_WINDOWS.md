# How to install Python 3.9 via Microsoft Store

1. Open the Microsoft Store app on your Windows 10/11 device. You can search for it in the Start menu or press the Windows logo key + S and type "store".
2. In the Microsoft Store app, search for "Python 3.9" or click on this link: [Python 3.9].
3. On the Python 3.9 app page, click on the "Get" button to download and install it. You may need to sign in with your Microsoft account if you haven't already.
4. Once the installation is complete, you can launch Python 3.9 from the Start menu or by typing "python" in the command prompt.
5. To verify that Python 3.9 is installed correctly, you can run the following command in the command prompt: `python --version`. It should display something like "Python 3.9.0".
6. Congratulations, you have successfully installed Python 3.9 via Microsoft Store. You can now use it to create and run Python scripts, modules, and applications. 🐍

# How to install Visual Studio Code via Microsoft Store

1. Open the Microsoft Store app on your Windows 10/11 device. You can search for it in the Start menu or press the Windows logo key + S and type "store".
2. In the Microsoft Store app, search for "Visual Studio Code" or click on this link: [Visual Studio Code](https://apps.microsoft.com/detail/XP9KHM4BK9FZ7Q?hl=en-US&gl=US).
3. On the Visual Studio Code app page, click on the "Get" button to download and install it. You may need to sign in with your Microsoft account if you haven't already.
4. Once the installation is complete, you can launch Visual Studio Code from the Start menu.
5. Congratulations, you have successfully installed Visual Studio Code via Microsoft Store. You can now use it to edit and debug code in various languages and frameworks. 💻

# How to install OpenVoice from GitHub using Visual Studio Code terminal

1. Launch Visual Studio Code and open a folder where you want to store the openvoice project. You can use the File > Open Folder menu command.
2. Open the integrated terminal in Visual Studio Code by using the Terminal > New Terminal menu command. Make sure that the terminal shows the same folder name as the one you opened in the previous step. If not, you can use the cd command to change the directory.
3. Clone the openvoice repository from GitHub by running this command in the terminal: `git clone https://github.com/myshell-ai/OpenVoice.git`.
4. Create a Python environment for the openvoice project. Press F1 and select the Python: Create Environment command.
5. Choose venv as the environment type and Python 3.9 as the interpreter. The environment will be created in the ".venv" folder within your project.
6. VS Code will ask you if you want to install the dependencies that are in the requirements.txt in the openvoice folder. Confirm this with Yes. If the prompt does not appear, go to step 10.
7. Make sure that the venv is activated in the terminal. You can tell if the venv is activated by looking at the prefix of the terminal prompt. It should show the name of the venv in parentheses, e.g. "(.venv)". If the venv is not activated, go to step 8. If the venv is activated, go to step 11.
8. Activate the venv manually by running this command in the terminal: `.\.venv\Scripts\activate`. Or create a new terminal in VS Code, which will activate the venv automatically for you.
9. Install ipykernel to run the notebooks. VS Code may prompt you to do this automatically. Confirm this with Yes. If not, install ipykernel manually by running this command in the terminal: `pip install ipykernel`.
10. Install the dependencies manually by running this command in the terminal: `pip install -r requirements.txt`. This step is only necessary if the dependencies were not installed automatically in step 6.
11. Download the required model checkpoints from this link: [Checkpoints](https://myshell-public-repo-hosting.s3.amazonaws.com/checkpoints_1226.zip), extract the zip and place the checkpoints folder in the openvoice folder.
12. To use openvoice, you can open and run the Jupyter notebooks that are in the openvoice folder. These notebooks will show you how to use openvoice to clone voices in various languages and styles. You can open the notebooks in VS Code by clicking on them in the Explorer on the left side.
13. To run the notebooks, you will need to install the Jupyter extension for VS Code. You can do this by pressing F1 and selecting the Extensions: Install Extensions command. Search for "Jupyter" and install the extension by Microsoft.
14. After installing the extension, you can run the notebooks by clicking on the play button next to the code blocks. The first notebook you should run is demo_part1.ipynb, which will guide you through the basic steps of using openvoice. You will need to provide a reference audio file for tone color extraction. You can use the sample audio files that are in the openvoice/resources folder or use your own audio file which you can also place inside the recources folder. The notebook will show you how to load the audio file, extract the tone color, and synthesize a new voice with the same tone color. You can also run the other notebooks, such as demo_part2.ipynb to see more advanced features of openvoice, such as style transfer and voice cloning. If you get a message that says something about ipywidgets being outdated or missing, you can install ipywidgets by running this command in the terminal: `pip install ipywidgets`.
15. Congratulations, you have successfully installed OpenVoice from GitHub using Visual Studio Code terminal. You can now use it to clone voices in various languages and styles. 🎙️
