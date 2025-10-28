# fastdeploy/input/kimi_k2_tokenizer.py

import os
import tiktoken
from shutil import copyfile
from typing import Dict, List, Optional, Tuple

from paddleformers.transformers import PretrainedTokenizer
from paddleformers.utils.log import logger
from tiktoken.load import load_tiktoken_bpe
# 新增导入
from tokenizers import AddedToken

class KimiK2Tokenizer(PretrainedTokenizer):
    """
    KimiK2 Tokenizer based on tiktoken.
    Adapted for the FastDeploy (PaddlePaddle) ecosystem to work without `trust_remote_code=True`.
    """
    
    resource_files_names = {"vocab_file": "tiktoken.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        added_tokens_decoder=None,
        additional_special_tokens=None,
        **kwargs,
    ):
        
        # --- 核心修正：适配 AddedToken 对象 ---
        special_tokens_map = {}
        if added_tokens_decoder:
            for token_id_str, token_info in added_tokens_decoder.items():
                token_id = int(token_id_str)
                # 检查 token_info 的类型
                if isinstance(token_info, AddedToken):
                    # 如果是 AddedToken 对象，通过 .content 属性访问
                    token_content = token_info.content
                elif isinstance(token_info, dict):
                    # 如果是字典，通过 ["content"] 键访问
                    token_content = token_info["content"]
                else:
                    # 处理未知类型，以防万一
                    raise TypeError(f"Unexpected type for token_info: {type(token_info)}")
                
                special_tokens_map[token_content] = token_id
        
        self.vocab_file = vocab_file
        pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        mergeable_ranks = load_tiktoken_bpe(vocab_file)

        self.model = tiktoken.Encoding(
            name=os.path.basename(vocab_file),
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens_map,
        )

        logger.info(f"Successfully loaded KimiK2 tiktoken model from {vocab_file} with {len(special_tokens_map)} special tokens.")
        
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            added_tokens_decoder=added_tokens_decoder,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )


    @property
    def vocab_size(self) -> int:
        return self.model.n_vocab

    def get_vocab(self) -> Dict[str, int]:
        vocab = self.model.special_tokens_set
        return {token: self.model.encode_single_token(token) for token in vocab}

    def _tokenize(self, text: str) -> List[str]:
        token_ids = self.model.encode(text, allowed_special="all")
        return [self._convert_id_to_token(token_id) for token_id in token_ids]

    def _convert_token_to_id(self, token: str) -> int:
        try:
            return self.model.encode_single_token(token)
        except KeyError:
            return self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        try:
            return self.model.decode_single_token_bytes(index).decode("utf-8", errors="replace")
        except KeyError:
            return self.unk_token

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        return text

    def decode(self, token_ids: List[int], **kwargs):
        # 移除可能由 paddleformers 基类传入的、tiktoken 不支持的参数
        kwargs.pop("skip_special_tokens", None)
        kwargs.pop("clean_up_tokenization_spaces", None)
        return self.model.decode(token_ids)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + self.resource_files_names["vocab_file"],
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        return (out_vocab_file,)