from .nodes.merge_tag_node import MergeTagNode
from .nodes.clear_tag_node import ClearTagNode
from .nodes.add_tag_node import AddTagNode
from .nodes.move_tag_to_top_node import MoveTagToTopNode
from .nodes.was import LoadImagesPairBatchNode, SaveImagesPairNode
NODE_CLASS_MAPPINGS = {
    "Merge Tag": MergeTagNode,
    "Clear Tag": ClearTagNode,
    "Add Tag": AddTagNode,
    "Move Tag To Top": MoveTagToTopNode,
    "Load Images Pair Batch": LoadImagesPairBatchNode,
    "Save Images Pair": SaveImagesPairNode
    }
    
print("\033[34mComfyUI Aimidi Nodes: \033[92mLoaded\033[0m")    