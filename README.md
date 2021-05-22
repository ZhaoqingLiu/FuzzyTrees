# Fuzzy Decision Trees
An algorithm framework integrating fuzzy decision trees and fuzzy ensemble trees.

##
**What to write in README.md?**
1. What the software is used to do: describe the information, basic functions;
2. How to run the code: environment installation, starting commands, etc.
3. How to use the software (briefly): instructions;
4. A description of the project directory structure, or a more detailed description of the software's rationale;
5. Frequently Asked Questions.

**The description of the project directory structure.**
- core/: Contains the business logic related code
- api/: Contains the interface file, which is used to provide data operations for the business logic.
- db/: Contains files related to the operation of the database, mainly used to interact with the database
- lib/: Contains custom modules commonly used in programs
- conf/: Contains the configuration file
- run.py: The startup file for your program. It is usually placed in the root directory of your project, because by default, the directory in which you run the file will be the first path in sys.path, eliminating the need to deal with environment variables.
- setup.py: The script for installing, deploying, and packaging. The industry standard is to use a popular Python packaging tool "setuptools" to develop this file.
- requires.txt: Contains a list of external thirty-party Python libraries that the software depends on. Each line in this file represents a package dependency, usually in a format like numpy>=0.90 to be recognized by the PIP. Users can simply install all dependencies using "pip install-r requires.txt".
- README: The project specification file.
##


