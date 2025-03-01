# Contributing to PerFlow-AI
Thank you for your interest in contributing to PerFlow-AI! Our community is open to everyone and welcomes all kinds of contributions, no matter how small or large. There are several ways you can contribute to the project:

- Identify and report any issues or bugs.

- Request or add support for a new analysis scenario.

- Suggest or implement new features.

- Improve documentation or contribute a how-to guide.

We also believe in the power of community support; thus, answering queries, offering PR reviews, and assisting others are also highly regarded and beneficial contributions.

Finally, one of the most impactful ways to support us is by raising awareness about PerFlow-AI. Talk about it in your blog posts and highlight how it’s driving your incredible projects. Express your support on social media if you’re using PerFlow-AI, or simply offer your appreciation by starring our repository!


## License
See LICENSE.


## Test
```
pip install -r requirements-dev.txt

# Linting, formatting and static type checking
pre-commit install

# You can manually run pre-commit with
pre-commit run --all-files

# Unit tests
pytest tests/
```


## Pull Requests & Code Reviews
Thank you for your contribution to PerFlow-AI! Before submitting the pull request, please ensure the PR meets the following criteria. This helps PerFlow-AI maintain the code quality and improve the efficiency of the review process.

PR Title and Classification
Only specific types of PRs will be reviewed. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:

- [Bugfix] for bug fixes.

- [CI/Build] for build or continuous integration improvements.

- [Docs] for documentation fixes and improvements.

- [Feat] for adding a new feature. Feature infomation should appear in the title.

- [Perf] for performance optimization.

- [Refactor] for code modifications that do not fix bug or add new features.

- [Style] for format modification.

- [Test] for adding new tests or modifying existing tests.

- [Misc] for PRs that do not fit the above categories. Please use this sparingly.

## Code Quality
The PR needs to meet the following code quality standards:
[Google Python style guide](https://google.github.io/styleguide/pyguide.html) and [Google C++ style guide](https://google.github.io/styleguide/cppguide.html).guide and Google C++ style guide.

Pass all linter checks. Please use `pre-commit` to format your code. See [https://pre-commit.com/#usage](https://pre-commit.com/#usage) if `pre-commit` is new to you.

The code needs to be well-documented to ensure future contributors can easily understand the code.

Include sufficient tests to ensure the project stays correct and robust. This includes both unit tests and integration tests.

Please add documentation to `docs/guide/` if the PR modifies the user-facing behaviors of PerFlow-AI. It helps PerFlow-AI users understand and utilize the new features or changes.

## Thank You
Finally, thank you for taking the time to read these guidelines and for your interest in contributing to PerFlow-AI. All of your contributions help make PerFlow-AI a great tool and community for everyone!
