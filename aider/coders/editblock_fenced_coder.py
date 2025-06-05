from ..dump import dump  # noqa: F401
from .editblock_coder import EditBlockCoder
from .editblock_fenced_prompts import EditBlockFencedPrompts


class EditBlockFencedCoder(EditBlockCoder):
    """A coder that uses fenced search/replace blocks for code modifications."""

    edit_format = "diff-fenced"
    gpt_prompts = EditBlockFencedPrompts()

    def __init__(self, main_model, io, **kwargs):
        # Handle editor_mode parameter for simplified prompts
        editor_mode = kwargs.pop('editor_mode', False)
        if editor_mode:
            from .editor_diff_fenced_prompts import EditorDiffFencedPrompts
            self.edit_format = "editor-diff-fenced"
            self.gpt_prompts = EditorDiffFencedPrompts()
        
        super().__init__(main_model, io, **kwargs)
