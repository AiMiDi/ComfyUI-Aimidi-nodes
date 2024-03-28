class ClearTagNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    },
                "optional": {       
                    "tag_text": ("STRING", {"forceInput": True}),
                    }
                }
               
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "clear_tag_node"
    CATEGORY = "Aimidi Nodes"

    def clear_tag_node(self, tag_text):
        tag_text = tag_text.split(",")
        tag_text = [tag.strip() for tag in tag_text]
        tag_text = [tag.replace(".", "") 
               .lower() if tag[0].isupper() and tag[1:].islower() else tag  
               for tag in tag_text if tag]
        tag_text = ", ".join(tag_text)
        return (tag_text,)
