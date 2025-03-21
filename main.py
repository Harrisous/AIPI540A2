#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script to run the project.
This script serves as the entry point for the project.
"""

import os
import sys
import subprocess

def main():
    """
    Main function to run the project.
    This launches the Streamlit app that serves as the user interface.
    """
    print("Starting ReviewAssist: Google Play Store Review Analyzer...")
    
    # Run the Streamlit app
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Streamlit not found. Please install required dependencies with: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main() 