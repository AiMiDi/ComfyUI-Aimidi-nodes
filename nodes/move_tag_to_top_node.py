class MoveTagToTopNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "move_tag_number": ("INT", {"default":1, "min":1, "step":1}),
                    "move_tag": ("STRING", {"multiline": True, "default": ""}),
                    },
                "optional": {       
                    "tag_text": ("STRING", {"forceInput": True}),
                    }
                }
               
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "move_tag_to_top"
    CATEGORY = "Aimidi Nodes"

    def move_tag_to_top(self, tag_text, move_tag_number, move_tag):
        tag_text = tag_text.split(",")
        move_tag = move_tag.split(",")
        tag_text = [tag.strip() for tag in tag_text]
        move_tag = [tag.strip() for tag in move_tag]
        move_count = 0
        for tag in move_tag:
            if(tag in tag_text):
                tag_text.remove(tag)
                tag_text.insert(0, tag)
                move_count += 1
            if move_count >= move_tag_number:
                break
        tag_text = ", ".join(tag_text)
        return (tag_text,)
