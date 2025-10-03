# Contributing to the Project

First off, thank you for considering contributing to this project! Any contributions you make are **greatly appreciated**.

This document provides guidelines for contributing to this project. Please read it carefully to ensure a smooth and effective contribution process.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Styleguides](#styleguides)
  - [Git Commit Messages](#git-commit-messages)
  - [Python Styleguide](#python-styleguide)

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please ensure the bug was not already reported by searching on GitHub under [Issues](httpss://github.com/arya2004/infenion-hackathon/issues).

If you're unable to find an open issue addressing the problem, [open a new one](httpss://github.com/arya2004/infenion-hackathon/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements

If you have an idea for an enhancement, please ensure it has not already been suggested by searching on GitHub under [Issues](httpss://github.com/arya2004/infenion-hackathon/issues).

If you're unable to find an open issue, [open a new one](httpss://github.com/arya2004/infenion-hackathon/issues/new). Please provide a clear description of the enhancement and its potential benefits.

### Pull Requests

1.  **Fork the repository** and create your branch from `main`.
2.  **Set up your environment** by installing the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Make your changes**.
4.  **Update the `README.md`** with details of changes to the interface, if applicable.
5.  **Ensure your code lints**. We use `black` for code formatting.
    ```bash
    pip install black
    black .
    ```
6.  **Commit your changes** using a clear and descriptive commit message.
7.  **Push to your fork** and submit a pull request to the `main` branch of the upstream repository.

## Styleguides

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature").
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
- Limit the first line to 72 characters or less.
- Reference issues and pull requests liberally after the first line.

### Python Styleguide

- All Python code should adhere to [PEP 8](httpss://www.python.org/dev/peps/pep-0008/).
- We use `black` for automatic code formatting to ensure consistency.
