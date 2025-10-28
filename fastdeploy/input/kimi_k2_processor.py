# fastdeploy/input/kimi_k2_processor.py

from fastdeploy.input.text_processor import DataProcessor
from fastdeploy.input.kimi_k2_tokenizer import KimiK2Tokenizer

class KimiK2Processor(DataProcessor):
    """
    Data processor for KimiK2 models. It overrides the tokenizer loading
    method to use the custom KimiK2Tokenizer.
    """

    def _load_tokenizer(self):
        """
        Overrides the default tokenizer loading to instantiate our custom
        KimiK2Tokenizer, avoiding the need for `trust_remote_code=True`.
        """
        return KimiK2Tokenizer.from_pretrained(self.model_name_or_path)