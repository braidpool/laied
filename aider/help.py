#!/usr/bin/env python

from aider.dump import dump  # noqa: F401


class Help:
    def __init__(self):
        """Simplified help system without local website dependency."""
        pass

    def ask(self, question):
        """Return a simple message directing users to online documentation."""
        return f"""# Question: {question}

For help with this question, please visit the online documentation at:
https://aider.chat/docs/

You can also:
- Join our Discord community: https://discord.gg/Y7X7bhMQFV
- Search GitHub issues: https://github.com/Aider-AI/aider/issues
- Check the FAQ: https://aider.chat/docs/faq.html
"""
