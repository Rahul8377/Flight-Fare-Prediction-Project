from setuptools import find_packages,setup
from typing import List

REQUIREMENT_FILE_NAME="requirements.txt"
HYPHEN_E_DOT = "-e ."

# get_requirements() function will provide list of library name
def get_requirements()->List[str]:

    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
    requirement_list = [reqirement_name.replace("\n", "") for reqirement_name in requirement_list]

    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)

    return requirement_list
    
setup(
    name= "flight",
    version= "0.0.1",
    author= "rahul",
    author_email= "rahul.amberia83@gmail.com",
    packages= find_packages(), # find_packages() will find sensor folder because in contains __init__.py
    install_requires = get_requirements()
)