class ReserveTagNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "reserve_tag_number": ("INT", {"default":1, "min":1, "step":1}),
                    "reserve_tag": ("STRING", {"multiline": True, "default": ""}),
                    },
                "optional": {       
                    "tag_text": ("STRING", {"forceInput": True}),
                    }
                }
               
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "reserve_tag"
    CATEGORY = "Aimidi Nodes"

    def reserve_tag(self, tag_text, reserve_tag_number, reserve_tag):
        tag_text = tag_text.split(",")
        reserve_tag = reserve_tag.split(",")
        tag_text = [tag.strip() for tag in tag_text]
        reserve_tag = [tag.strip() for tag in reserve_tag]
        reserved_count = 0
        new_tags = []
        for tag in reserve_tag:
            if(tag in tag_text):
                new_tags.insert(0, tag)
                reserved_count += 1
            if reserved_count >= reserve_tag_number:
                break
        tag_text = ", ".join(new_tags)
        return (tag_text,)
