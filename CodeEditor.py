# CodeEditor -- an AST and RAG based repository map editor
#
# See Aider's aider/repomap.py for something similar
#
# The point of this module is that it handles file-level scope, so that:
# 1. We don't have to give LLMs entire files, only what matters for the problem
#   at hand. Therefore we can minimize context usage.
# 2. The CodeEditor decides where and now files are written, not the LLM. If the
#   LLM wants to create class foo::bar, this module decides whether to create
#   class bar in a file foo.py or class bar in foo/__init__.py etc.
# 3. This completely solves the problem of AI's making correctly formatted diffs
#   (which aider has a problem with) Making diffs is an unnecessary task if
#   instead the LLM can just say "Add a function foo".
# 4. This module should be smart about using git. For instance if we have a file
#   foo.py containing a class Bar, and that begins to expand in scope, it should
#   use `git mv foo.py foo/__init__.py` so that edit history is preserved.

class CodeEditor:
    def __init__():
        """ This constructor should scan the codebase using git to build an AST
            or GraphRAG of the code.
        """
        pass

    def get(name, scope) -> str:
        """
            Get the code for an object. `name` is assumed to be unambiguous.
            e.g. in python foo::bar should be resolvable within `scope`.
            There may be better ways to specify this...

            This should also be able to get interface definitions for
            dependencies, even though we can't edit them.
        """
        pass

    def put(name, scope, code) -> bool:
        """
            Write (or re-write) an object (function or class) and replace it
            with `code`.
        """
        pass

    def affects(name) -> list[str]:
        """
            If we change `name`, what else does it affect? This should return a
            list of functions, modules, or classes that use this name or import
            this module. Here we should be using the GraphRAG.
        """
        pass

    def interface(name) -> str:
        """
            Return the interface for a module or class. This is a list of
            variables and methods in the module or class, their types, and
            function calling interface.

            We must orchestrate the LLM to call this function before beginning
            to write code that uses a module, class or function.
        """
        pass

    def add_dependency(name) -> bool:
        """
            Add a dependency. This should edit requirements.txt, pyproject.toml,
            Cargo.toml or whatever is required by the language. It should pull
            the code for the dependency if necessary.

            This will generally be followed by a call to
            `CodeEditor::interface()` so that it can be used.

            We will want to orchestrate the LLM so that before it starts writing
            any code involving a dependency, it calls this function and
            `CodeEditor::interface()` to prevent hallucinations.
        """
        pass
