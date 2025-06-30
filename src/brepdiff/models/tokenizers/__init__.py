from brepdiff.models.tokenizers.base import Tokenizer, Tokens
from brepdiff.models.tokenizers.pre_mask_uv_tokenizer_3d import PreMaskUvTokenizer3D

TOKENIZERS = {
    PreMaskUvTokenizer3D.name: PreMaskUvTokenizer3D,
}
