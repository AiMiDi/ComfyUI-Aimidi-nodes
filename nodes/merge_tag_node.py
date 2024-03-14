class MergeTagNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    },
                "optional": {
                    "tag_text_a": ("STRING", {"forceInput": True}),
                    "tag_text_b": ("STRING", {"forceInput": True}),
                    "tag_text_c": ("STRING", {"forceInput": True}),
                    "tag_text_d": ("STRING", {"forceInput": True}),
                    }
                }
               
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "merge_tag_node"
    CATEGORY = "Aimidi Nodes"

    def merge_tag_node(self, **kwargs):
        text_inputs: list[str] = []
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            text_inputs += v.split(",")
        # Strip spaces from each tag and remove duplicates by converting to a set and back to list
        TagText = list(set(tag.strip() for tag in text_inputs))
        TagText = ", ".join(TagText)
        return (TagText,)
