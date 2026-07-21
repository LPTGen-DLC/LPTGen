# Prompt Descriptions: Seed Generation and Predicate-Guided Mutation

This file documents the two core prompt templates used by HARLO for operator-level test generation.

## 1. Common Lowering-Path Seed Generation

```text
Generate {config_size} diverse API parameter configurations for operator "{operator_name}" strictly following these requirements.

1. Parameter types
- Tensor inputs must be represented as [shape, dtype] tuples, for example ([1, 3, 224, 224], "float32").
- Non-tensor parameters must be concrete Python literals or structured values.

2. Diversity requirements
- Cover both common and edge-case inputs.
- Vary tensor ranks, tensor shapes, and data types.
- Include valid non-default parameter combinations when possible.

3. Constraints
- Only use parameters defined by this signature:
{operator_signature}
- Only generate values consistent with the following documentation or retrieved constraints:
{operator_constraints}
- Do not produce invalid parameter combinations.
- Do not use placeholder text.
- Do not include explanations.

4. Output format
- Return a Python list of {config_size} dictionaries.
- Each dictionary must map parameter names to values directly.

Output only the Python list.
```

## 2. Predicate-Guided Corner-Case Mutation

```text
Generate {config_size} mutated parameter configurations for operator "{operator_name}" from the following seed configuration:
{seed_configuration}

The goal is to explore a lowering path controlled by this predicate or its negation:
{target_predicate}

Mutation requirements:
- Preserve the same parameter schema as the seed.
- Mutate only parameters relevant to the predicate when possible.
- Keep unrelated assignments stable unless a coordinated mutation is required.
- Respect inter-parameter constraints and type constraints.
- Produce valid configurations that are executable.

Operator signature:
{operator_signature}

Output requirements:
- Return a Python list of {config_size} dictionaries.
- Do not include explanations, comments, or markdown.
- Output only the Python list.
```
