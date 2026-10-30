"""Summarization prompts for the multilingual loop types.

Each template is the standard loop's summarization prompt (see
``EvaluationConfig.get_task_configs``) with two changes: the model is asked to
write the sentence in the target language, and the fixed opening phrase
"write a python function to" is replaced by a translation in that language.
The generation prompt, the assert-line postprocess and the test-based failure
condition stay the same as in the standard loop.
"""

# Opening phrase the summary must start with, in simple wording
SUMMARY_OPENINGS = {
    "hindi": "एक पायथन फ़ंक्शन लिखें जो",  # "write a Python function that"
    "spanish": "escribe una función en Python para",  # "write a function in Python to"
}

# Loop type name -> summarization prompt template ({code} is filled in by the task)
MULTILINGUAL_SUMMARIZATION_TEMPLATES = {
    "summarization_hindi": (
        "Use one sentence in Hindi to summarize the following code and start with "
        f"'{SUMMARY_OPENINGS['hindi']}':\n\n"
        "```\n{code}\n```\n\n"
        f"```\n{SUMMARY_OPENINGS['hindi']}\n```"
    ),
    "summarization_spanish": (
        "Use one sentence in Spanish to summarize the following code and start with "
        f"'{SUMMARY_OPENINGS['spanish']}':\n\n"
        "```\n{code}\n```\n\n"
        f"```\n{SUMMARY_OPENINGS['spanish']}\n```"
    ),
}
