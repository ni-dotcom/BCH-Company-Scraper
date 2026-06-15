import sys
!{sys.executable} -m pip install selenium
!apt-get update
# Install Google Chrome stable via .deb package
!wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
!dpkg -i google-chrome-stable_current_amd64.deb
!apt-get install -f # Fix broken dependencies if any


import os
import requests # Needed for robust ChromeDriver version detection

# Get the installed google-chrome-stable version
print("Detecting Google Chrome major version...")
chrome_version_output = !google-chrome-stable --version
print(f"Raw google-chrome-stable --version output: {chrome_version_output}")

chrome_major_version = None
if chrome_version_output and len(chrome_version_output) > 0:
    try:
        # Expected format: 'Google Chrome X.Y.Z.W'
        version_string = chrome_version_output[0].split(' ')[2]
        chrome_major_version = version_string.split('.')[0]
        print(f"Detected Google Chrome major version: {chrome_major_version}")
    except Exception as e:
        print(f"Error parsing google-chrome-stable --version output: {e}")

if not chrome_major_version:
    print("Critical Error: Unable to determine Google Chrome version. This will likely cause ChromeDriver installation to fail.")
    print("Falling back to a common Chrome major version (e.g., 115) for ChromeDriver download as a last resort.")
    chrome_major_version = "115" # A common stable version for Colab environments


# Now, find the compatible ChromeDriver version and download it.
print(f"Finding compatible ChromeDriver for Chrome major version {chrome_major_version}...")
try:
    # Use the Chrome for Testing JSON API to find the latest compatible ChromeDriver URL
    response = requests.get("https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json")
    response.raise_for_status() # Raise an exception for HTTP errors
    versions_data = response.json()

    chromedriver_url = None
    # Iterate through channels to find a ChromeDriver that matches the detected Chrome major version
    # The API structure has changed, accessing downloads directly via channels
    for channel_name in ['Stable', 'Beta', 'Dev', 'Canary']: # Prioritize Stable, then Beta, etc.
        if channel_name in versions_data['channels']:
            channel_info = versions_data['channels'][channel_name]
            channel_chrome_version = channel_info['version'].split('.')[0] # Get major version of Chrome in this channel
            if channel_chrome_version == chrome_major_version:
                for download in channel_info['downloads']['chromedriver']:
                    if download['platform'] == 'linux64':
                        chromedriver_url = download['url']
                        break
            if chromedriver_url: # Found a compatible ChromeDriver in this channel
                break

    if chromedriver_url:
        print(f"Downloading ChromeDriver from: {chromedriver_url}")
        # Use a temporary file for downloading and then unzip
        !wget -N "$chromedriver_url" -O chromedriver_temp.zip
        !unzip -o chromedriver_temp.zip -d /tmp/chromedriver_extracted/
        # The extracted structure is usually `chromedriver-linux64/chromedriver`
        # We need to move the actual executable.
        !mv /tmp/chromedriver_extracted/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver
        !chmod +x /usr/local/bin/chromedriver
        print("ChromeDriver installed and configured successfully.")
        # Set the global CHROMEDRIVER_PATH to the new location
        CHROMEDRIVER_PATH = '/usr/local/bin/chromedriver'
    else:
        print(f"Error: Could not find a compatible ChromeDriver URL for Chrome major version {chrome_major_version} across all channels.")
        print("Falling back to a default ChromeDriver path (might not be compatible).")
        CHROMEDRIVER_PATH = '/usr/lib/chromium-browser/chromedriver'

except requests.exceptions.RequestException as e:
    print(f"Error fetching Chrome for Testing versions JSON: {e}")
    print("Falling back to a default ChromeDriver path (might not be compatible).")
    CHROMEDRIVER_PATH = '/usr/lib/chromium-browser/chromedriver'
except Exception as e:
    print(f"An unexpected error occurred during ChromeDriver download: {e}")
    print("Falling back to a default ChromeDriver path (might not be compatible).")
    CHROMEDRIVER_PATH = '/usr/lib/chromium-browser/chromedriver'

print(f"Final CHROMEDRIVER_PATH set to: {CHROMEDRIVER_PATH}")



from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import os # Make sure os is imported for path handling

# Setup Chrome options for headless browsing
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# To avoid being detected as a bot
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
chrome_options.add_experimental_option('useAutomationExtension', False)
chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36')

# Path to the Chrome binary and ChromeDriver in Colab
# Update CHROME_PATH to 'google-chrome' as 'google-chrome-stable' installs it as 'google-chrome'
CHROME_PATH = '/usr/bin/google-chrome'
# CHROMEDRIVER_PATH should be set by the previous cell, but let's ensure it's used.
# If not already defined, we'll try a common path or rely on previous cell's output.
if 'CHROMEDRIVER_PATH' not in globals():
    CHROMEDRIVER_PATH = '/usr/local/bin/chromedriver' # Default if previous cell failed to set it
    print(f"Warning: CHROMEDRIVER_PATH was not set by the previous cell. Using default: {CHROMEDRIVER_PATH}")

# Explicitly set the binary location for Chrome
chrome_options.binary_location = CHROME_PATH

# Initialize WebDriver once globally
try:
    # Use Service object for executable_path
    service = Service(executable_path=CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    print("Selenium WebDriver initialized successfully.")
except Exception as e:
    print(f"Error initializing Selenium WebDriver: {e}")
    driver = None # Set driver to None if initialization fails
