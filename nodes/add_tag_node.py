class AddTagNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {       
                    "tag_text": ("STRING", {"forceInput": True}),
                    "add_tag": ("STRING", {"multiline": True, "default": ""}),
                    }
                }
               
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "add_tag"
    CATEGORY = "Aimidi Nodes"

    def add_tag(self, tag_text, add_tag):
        tag_text = tag_text.split(",")
        add_tag = add_tag.split(",")
        tag_text = [tag.strip() for tag in tag_text]
        add_tag = [tag.strip() for tag in add_tag]
        for tag in add_tag:
            if(tag in tag_text):
                tag_text.remove(tag)
            tag_text.insert(0, tag)
        tag_text = ", ".join(tag_text)
        return (tag_text,)
