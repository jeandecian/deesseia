# Contributing

We welcome contributions! To maintain quality, please follow these guidelines:

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <subject>

<type>:
  feat     - New feature
  fix      - Bug fix
  docs     - Documentation
  test     - Tests
  refactor - Code refactoring
  perf     - Performance
  chore    - Maintenance

<scope>:
  core, ml, nlp, dl, llm, cv, viz, ts, graph, utils, tests

Examples:
  feat(core): add DataLoader for CSV files
  fix(ml): handle edge case in regression
  docs(readme): update installation instructions
```

## SOLID Principles

All code must follow SOLID:

- **S**ingle Responsibility: Each module does one thing
- **O**pen/Closed: Open for extension, closed for modification
- **L**iskov Substitution: Subtypes must be substitutable
- **I**nterface Segregation: Small, focused interfaces
- **D**ependency Inversion: Depend on abstractions, not concretions

## Code Quality

- **Tests**: All features must have tests (pytest)
- **Types**: Use type hints (mypy must pass)
- **Style**: Follow black and isort
