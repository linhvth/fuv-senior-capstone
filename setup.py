"""
To install all necessary packages in Python to execute the program.

Linh Vu (2024)
"""
import subprocess # process installisation

PACKAGES = [] # global var

def getPackagesList(file_path):
    """
    Reads package names from a text file and store them as a list in 
    PACKAGES global variable defined above.

    Args:
        file_path (str): The path to the text file containing package names.

    Returns:
        list: A list of package names extracted from the file. 

    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                PACKAGES.append(line.strip())  # Remove leading/trailing whitespace
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    print(PACKAGES)


def upgrade_install_packages(filename):
    """
    Checks if a list of packages are installed and upgrades them to the latest version if necessary.
    If a package is not installed, it attempts to install it.

    Args:
        filename (.txt): A .txt file of package names to check and upgrade/install.

    Prints information about installed/upgraded/installed packages.
    """
    getPackagesList(filename)
    for package in PACKAGES:
        # Check if the package is already installed
        result = subprocess.run(["pip", "show", package], 
                                capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"{package} is already installed. Upgrading...")

            # Upgrade the package to the latest version
            subprocess.run(["pip", "install", "--upgrade", package])
        else:
            print(f"{package} is not installed. Installing...")

            # Install the package
            subprocess.run(["pip", "install", package])

if __name__ == "__main__":
    filename = 'packages.txt'
    getPackagesList(file_path=filename)
    upgrade_install_packages(PACKAGES)
