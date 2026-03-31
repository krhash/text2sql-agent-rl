"""PromptBuilder — schema-aware few-shot prompt construction."""
from __future__ import annotations

import json
from pathlib import Path


FEW_SHOT_EXAMPLES = """
### Example 1
Schema: stadium : Stadium_ID (number) , Location (text) , Name (text) , Capacity (number)
Question: How many stadiums are there?
SQL: <SQL_START> SELECT count(*) FROM stadium <SQL_END>

### Example 2
Schema: singer : Singer_ID (number) , Name (text) , Country (text) , Age (number)
Question: What are the names of all singers from France?
SQL: <SQL_START> SELECT Name FROM singer WHERE Country = 'France' <SQL_END>

### Example 3
Schema: employee : Employee_ID (number) , Name (text) , Department (text) , Salary (number)
Question: What is the average salary of employees in each department?
SQL: <SQL_START> SELECT Department, avg(Salary) FROM employee GROUP BY Department <SQL_END>
"""

DEFAULT_SYSTEM = (
    "You are a SQL expert. Given a database schema and a question, "
    "generate the correct SQL query.\n\n"
    "You MUST follow these output rules EXACTLY:\n"
    "- You MUST wrap your SQL in tags: <SQL_START> your sql here <SQL_END>\n"
    "- You MUST output ONLY the tags and SQL — NO other text\n"
    "- You MUST NOT add explanations, comments, or markdown\n"
    "- You MUST output a SINGLE SQL statement with NO semicolon\n\n"
    "Example: <SQL_START> SELECT col FROM table WHERE condition <SQL_END>"
)


class PromptBuilder:
    """
    Builds schema-aware few-shot prompts for text-to-SQL inference.

    Serialisable to/from JSON so prompt optimization can save and reload
    the best prompt across stages.
    """

    def __init__(self, schema_path: str,
                 system_prompt: str = DEFAULT_SYSTEM,
                 few_shot_examples: str = FEW_SHOT_EXAMPLES):
        with open(schema_path) as f:
            schemas = json.load(f)
        self.schema_dict       = {s['db_id']: s for s in schemas}
        self.schema_path       = schema_path
        self.system_prompt     = system_prompt
        self.few_shot_examples = few_shot_examples

    def build(self, question: str, db_id: str) -> str:
        s = self.schema_dict[db_id]
        return (
            f"{self.system_prompt}\n\n"
            f"Here are some examples:\n{self.few_shot_examples}\n"
            "Now answer the following:\n\n"
            f"### Database Schema\n{s['Schema (values (type))']}\n\n"
            f"### Primary Keys\n{s['Primary Keys']}\n\n"
            f"### Foreign Keys\n{s['Foreign Keys']}\n\n"
            f"### Question\n{question}\n\n"
            "### SQL\n"
        )

    def with_system(self, new_system: str) -> "PromptBuilder":
        """Return a new PromptBuilder with a different system prompt. Immutable."""
        pb = PromptBuilder.__new__(PromptBuilder)
        pb.schema_dict       = self.schema_dict
        pb.schema_path       = self.schema_path
        pb.system_prompt     = new_system
        pb.few_shot_examples = self.few_shot_examples
        return pb

    def with_few_shot(self, new_few_shot: str) -> "PromptBuilder":
        """Return a new PromptBuilder with different few-shot examples."""
        pb = PromptBuilder.__new__(PromptBuilder)
        pb.schema_dict       = self.schema_dict
        pb.schema_path       = self.schema_path
        pb.system_prompt     = self.system_prompt
        pb.few_shot_examples = new_few_shot
        return pb

    def save(self, path: Path | str):
        """Serialise to JSON for the prompt optimisation → infer handoff."""
        data = {
            "schema_path"      : self.schema_path,
            "system_prompt"    : self.system_prompt,
            "few_shot_examples": self.few_shot_examples,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def from_file(cls, path: Path | str) -> "PromptBuilder":
        """Load a saved PromptBuilder from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            schema_path       = data["schema_path"],
            system_prompt     = data["system_prompt"],
            few_shot_examples = data["few_shot_examples"],
        )
