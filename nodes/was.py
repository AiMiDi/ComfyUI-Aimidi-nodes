# copy from WAS Suite

import hashlib
import os
import re
import json
import time
import socket
import numpy as np
import torch
import glob
import random
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import comfy.model_management
import folder_paths as comfy_paths

NODE_FILE = os.path.abspath(__file__)
WAS_SUITE_ROOT = os.path.dirname(NODE_FILE)
WAS_CONFIG_DIR = os.environ.get('WAS_CONFIG_DIR', WAS_SUITE_ROOT)
WAS_DATABASE = os.path.join(WAS_CONFIG_DIR, 'was_suite_settings.json')
WAS_HISTORY_DATABASE = os.path.join(WAS_CONFIG_DIR, 'was_history.json')
WAS_CONFIG_FILE = os.path.join(WAS_CONFIG_DIR, 'was_suite_config.json')
ALLOWED_EXT = ('.jpeg', '.jpg', '.png',
                        '.tiff', '.gif', '.bmp', '.webp')

class cstr(str):
    class color:
        END = '\33[0m'
        BOLD = '\33[1m'
        ITALIC = '\33[3m'
        UNDERLINE = '\33[4m'
        BLINK = '\33[5m'
        BLINK2 = '\33[6m'
        SELECTED = '\33[7m'

        BLACK = '\33[30m'
        RED = '\33[31m'
        GREEN = '\33[32m'
        YELLOW = '\33[33m'
        BLUE = '\33[34m'
        VIOLET = '\33[35m'
        BEIGE = '\33[36m'
        WHITE = '\33[37m'

        BLACKBG = '\33[40m'
        REDBG = '\33[41m'
        GREENBG = '\33[42m'
        YELLOWBG = '\33[43m'
        BLUEBG = '\33[44m'
        VIOLETBG = '\33[45m'
        BEIGEBG = '\33[46m'
        WHITEBG = '\33[47m'

        GREY = '\33[90m'
        LIGHTRED = '\33[91m'
        LIGHTGREEN = '\33[92m'
        LIGHTYELLOW = '\33[93m'
        LIGHTBLUE = '\33[94m'
        LIGHTVIOLET = '\33[95m'
        LIGHTBEIGE = '\33[96m'
        LIGHTWHITE = '\33[97m'

        GREYBG = '\33[100m'
        LIGHTREDBG = '\33[101m'
        LIGHTGREENBG = '\33[102m'
        LIGHTYELLOWBG = '\33[103m'
        LIGHTBLUEBG = '\33[104m'
        LIGHTVIOLETBG = '\33[105m'
        LIGHTBEIGEBG = '\33[106m'
        LIGHTWHITEBG = '\33[107m'

        @staticmethod
        def add_code(name, code):
            if not hasattr(cstr.color, name.upper()):
                setattr(cstr.color, name.upper(), code)
            else:
                raise ValueError(f"'cstr' object already contains a code with the name '{name}'.")

    def __new__(cls, text):
        return super().__new__(cls, text)

    def __getattr__(self, attr):
        if attr.lower().startswith("_cstr"):
            code = getattr(self.color, attr.upper().lstrip("_cstr"))
            modified_text = self.replace(f"__{attr[1:]}__", f"{code}")
            return cstr(modified_text)
        elif attr.upper() in dir(self.color):
            code = getattr(self.color, attr.upper())
            modified_text = f"{code}{self}{self.color.END}"
            return cstr(modified_text)
        elif attr.lower() in dir(cstr):
            return getattr(cstr, attr.lower())
        else:
            raise AttributeError(f"'cstr' object has no attribute '{attr}'")

    def print(self, **kwargs):
        print(self, **kwargs)

#! MESSAGE TEMPLATES
cstr.color.add_code("msg", f"{cstr.color.BLUE}WAS Node Suite: {cstr.color.END}")
cstr.color.add_code("warning", f"{cstr.color.BLUE}WAS Node Suite {cstr.color.LIGHTYELLOW}Warning: {cstr.color.END}")
cstr.color.add_code("error", f"{cstr.color.RED}WAS Node Suite {cstr.color.END}Error: {cstr.color.END}")

class WASDatabase:
    """
    The WAS Suite Database Class provides a simple key-value database that stores
    data in a flatfile using the JSON format. Each key-value pair is associated with
    a category.

    Attributes:
        filepath (str): The path to the JSON file where the data is stored.
        data (dict): The dictionary that holds the data read from the JSON file.

    Methods:
        insert(category, key, value): Inserts a key-value pair into the database
            under the specified category.
        get(category, key): Retrieves the value associated with the specified
            key and category from the database.
        update(category, key): Update a value associated with the specified
            key and category from the database.
        delete(category, key): Deletes the key-value pair associated with the
            specified key and category from the database.
        _save(): Saves the current state of the database to the JSON file.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        try:
            with open(filepath, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}

    def catExists(self, category):
        return category in self.data

    def keyExists(self, category, key):
        return category in self.data and key in self.data[category]

    def insert(self, category, key, value):
        if not isinstance(category, str) or not isinstance(key, str):
            cstr("Category and key must be strings").error.print()
            return

        if category not in self.data:
            self.data[category] = {}
        self.data[category][key] = value
        self._save()

    def update(self, category, key, value):
        if category in self.data and key in self.data[category]:
            self.data[category][key] = value
            self._save()

    def updateCat(self, category, dictionary):
        self.data[category].update(dictionary)
        self._save()

    def get(self, category, key):
        return self.data.get(category, {}).get(key, None)

    def getDB(self):
        return self.data

    def insertCat(self, category):
        if not isinstance(category, str):
            cstr("Category must be a string").error.print()
            return

        if category in self.data:
            cstr(f"The database category '{category}' already exists!").error.print()
            return
        self.data[category] = {}
        self._save()

    def getDict(self, category):
        if category not in self.data:
            cstr(f"The database category '{category}' does not exist!").error.print()
            return {}
        return self.data[category]

    def delete(self, category, key):
        if category in self.data and key in self.data[category]:
            del self.data[category][key]
            self._save()

    def _save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=4)
        except FileNotFoundError:
            cstr(f"Cannot save database to file '{self.filepath}'. "
                 "Storing the data in the object instead. Does the folder and node file have write permissions?").warning.print()
        except Exception as e:
            cstr(f"Error while saving JSON data: {e}").error.print()

# Initialize the settings database
WDB = WASDatabase(WAS_DATABASE)

# WAS Token Class

class TextTokens:
    def __init__(self):
        self.WDB = WDB
        if not self.WDB.getDB().__contains__('custom_tokens'):
            self.WDB.insertCat('custom_tokens')
        self.custom_tokens = self.WDB.getDict('custom_tokens')

        self.tokens = {
            '[time]': str(time.time()).replace('.','_'),
            '[hostname]': socket.gethostname(),
            '[cuda_device]': str(comfy.model_management.get_torch_device()),
            '[cuda_name]': str(comfy.model_management.get_torch_device_name(device=comfy.model_management.get_torch_device())),
        }

        if '.' in self.tokens['[time]']:
            self.tokens['[time]'] = self.tokens['[time]'].split('.')[0]

        try:
            self.tokens['[user]'] = os.getlogin() if os.getlogin() else 'null'
        except Exception:
            self.tokens['[user]'] = 'null'

    def addToken(self, name, value):
        self.custom_tokens.update({name: value})
        self._update()

    def removeToken(self, name):
        self.custom_tokens.pop(name)
        self._update()

    def format_time(self, format_code):
        return time.strftime(format_code, time.localtime(time.time()))

    def parseTokens(self, text):
        tokens = self.tokens.copy()
        if self.custom_tokens:
            tokens.update(self.custom_tokens)

        # Update time
        tokens['[time]'] = str(time.time())
        if '.' in tokens['[time]']:
            tokens['[time]'] = tokens['[time]'].split('.')[0]

        for token, value in tokens.items():
            if token.startswith('[time('):
                continue
            pattern = re.compile(re.escape(token))
            text = pattern.sub(value, text)

        def replace_custom_time(match):
            format_code = match.group(1)
            return self.format_time(format_code)

        text = re.sub(r'\[time\((.*?)\)\]', replace_custom_time, text)

        return text

    def _update(self):
        self.WDB.updateCat('custom_tokens', self.custom_tokens)

#! WAS SUITE CONFIG

was_conf_template = {
                    "run_requirements": True,
                    "suppress_uncomfy_warnings": True,
                    "show_startup_junk": True,
                    "show_inspiration_quote": True,
                    "text_nodes_type": "STRING",
                    "webui_styles": None,
                    "webui_styles_persistent_update": True,
                    "blip_model_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth",
                    "blip_model_vqa_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth",
                    "sam_model_vith_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    "sam_model_vitl_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    "sam_model_vitb_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    "history_display_limit": 36,
                    "use_legacy_ascii_text": False,
                    "ffmpeg_bin_path": "/path/to/ffmpeg",
                    "ffmpeg_extra_codecs": {
                        "avc1": ".mp4",
                        "h264": ".mkv",
                    },
                    "wildcards_path": os.path.join(WAS_SUITE_ROOT, "wildcards"),
                    "wildcard_api": True,
                }

# Create, Load, or Update Config

def getSuiteConfig():
    global was_conf_template
    try:
        with open(WAS_CONFIG_FILE, "r") as f:
            was_config = json.load(f)
    except OSError as e:
        cstr(f"Unable to load conf file at `{WAS_CONFIG_FILE}`. Using internal config template.").error.print()
        return was_conf_template
    except Exception as e:
        cstr(f"Unable to load conf file at `{WAS_CONFIG_FILE}`. Using internal config template.").error.print()
        return was_conf_template
    return was_config

def updateSuiteConfig(conf):
    try:
        with open(WAS_CONFIG_FILE, "w", encoding='utf-8') as f:
            json.dump(conf, f, indent=4)
    except OSError as e:
        print(e)
        return False
    except Exception as e:
        print(e)
        return False
    return True

if not os.path.exists(WAS_CONFIG_FILE):
    if updateSuiteConfig(was_conf_template):
        cstr(f'Created default conf file at `{WAS_CONFIG_FILE}`.').msg.print()
        was_config = getSuiteConfig()
    else:
        cstr(f"Unable to create default conf file at `{WAS_CONFIG_FILE}`. Using internal config template.").error.print()
        was_config = was_conf_template

else:
    was_config = getSuiteConfig()

    update_config = False
    for sett_ in was_conf_template.keys():
        if not was_config.__contains__(sett_):
            was_config.update({sett_: was_conf_template[sett_]})
            update_config = True

    if update_config:
        updateSuiteConfig(was_config)


# SET TEXT TYPE
TEXT_TYPE = "STRING"
if was_config and was_config.__contains__('text_nodes_type'):
    if was_config['text_nodes_type'].strip() != '':
        TEXT_TYPE = was_config['text_nodes_type'].strip()
if was_config and was_config.__contains__('use_legacy_ascii_text'):
    if was_config['use_legacy_ascii_text']:
        TEXT_TYPE = "ASCII"
        cstr("use_legacy_ascii_text is `True` in `was_suite_config.json`. `ASCII` type is deprecated and the default will be `STRING` in the future.").warning.print()

# Update image history

def update_history_images(new_paths):
    HDB = WASDatabase(WAS_HISTORY_DATABASE)
    if HDB.catExists("History") and HDB.keyExists("History", "Images"):
        saved_paths = HDB.get("History", "Images")
        for path_ in saved_paths:
            if not os.path.exists(path_):
                saved_paths.remove(path_)
        if isinstance(new_paths, str):
            if new_paths in saved_paths:
                saved_paths.remove(new_paths)
            saved_paths.append(new_paths)
        elif isinstance(new_paths, list):
            for path_ in new_paths:
                if path_ in saved_paths:
                    saved_paths.remove(path_)
                saved_paths.append(path_)
        HDB.update("History", "Images", saved_paths)
    else:
        if not HDB.catExists("History"):
            HDB.insertCat("History")
        if isinstance(new_paths, str):
            HDB.insert("History", "Images", [new_paths])
        elif isinstance(new_paths, list):
            HDB.insert("History", "Images", new_paths)

# Update output image history

def update_history_output_images(new_paths):
    HDB = WASDatabase(WAS_HISTORY_DATABASE)
    category = "Output_Images"
    if HDB.catExists("History") and HDB.keyExists("History", category):
        saved_paths = HDB.get("History", category)
        for path_ in saved_paths:
            if not os.path.exists(path_):
                saved_paths.remove(path_)
        if isinstance(new_paths, str):
            if new_paths in saved_paths:
                saved_paths.remove(new_paths)
            saved_paths.append(new_paths)
        elif isinstance(new_paths, list):
            for path_ in new_paths:
                if path_ in saved_paths:
                    saved_paths.remove(path_)
                saved_paths.append(path_)
        HDB.update("History", category, saved_paths)
    else:
        if not HDB.catExists("History"):
            HDB.insertCat("History")
        if isinstance(new_paths, str):
            HDB.insert("History", category, [new_paths])
        elif isinstance(new_paths, list):
            HDB.insert("History", category, new_paths)

# Update text file history

def update_history_text_files(new_paths):
    HDB = WASDatabase(WAS_HISTORY_DATABASE)
    if HDB.catExists("History") and HDB.keyExists("History", "TextFiles"):
        saved_paths = HDB.get("History", "TextFiles")
        for path_ in saved_paths:
            if not os.path.exists(path_):
                saved_paths.remove(path_)
        if isinstance(new_paths, str):
            if new_paths in saved_paths:
                saved_paths.remove(new_paths)
            saved_paths.append(new_paths)
        elif isinstance(new_paths, list):
            for path_ in new_paths:
                if path_ in saved_paths:
                    saved_paths.remove(path_)
                saved_paths.append(path_)
        HDB.update("History", "TextFiles", saved_paths)
    else:
        if not HDB.catExists("History"):
            HDB.insertCat("History")
        if isinstance(new_paths, str):
            HDB.insert("History", "TextFiles", [new_paths])
        elif isinstance(new_paths, list):
            HDB.insert("History", "TextFiles", new_paths)

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# SHA-256 Hash
def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

class LoadImagesPairBatchNode:
    def __init__(self):
        self.HDB = WASDatabase(WAS_HISTORY_DATABASE)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_image", "incremental_image", "random"],),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "label": ("STRING", {"default": 'Batch 001', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "pattern": ("STRING", {"default": '*', "multiline": False}),
                "allow_RGBA_output": (["false","true"],),
            },
            "optional": {
                "filename_text_extension": (["true", "false"],),
            }
        }

    RETURN_TYPES = ("IMAGE", TEXT_TYPE, TEXT_TYPE, TEXT_TYPE)
    RETURN_NAMES = ("image", "text", "image_filename", "image_filepath")
    FUNCTION = "load_batch_images"

    CATEGORY = "Aimidi Nodes"

    def load_txt_file(self, file_path=''):
        
        if not os.path.exists(file_path):
            return ''
        with open(file_path, 'r', encoding="utf-8", newline='\n') as file:
            text = file.read()

        # Write to file history
        update_history_text_files(file_path)

        import io
        lines = []
        for line in io.StringIO(text):
            if not line.strip().startswith('#'):
                if ( not line.strip().startswith("\n")
                    or not line.strip().startswith("\r")
                    or not line.strip().startswith("\r\n") ):
                    line = line.replace("\n", '').replace("\r",'').replace("\r\n",'')
                lines.append(line.replace("\n",'').replace("\r",'').replace("\r\n",''))
    
        return "\n".join(lines)
    
    def load_batch_images(self, path, pattern='*', index=0, mode="single_image", label='Batch 001', allow_RGBA_output='false', filename_text_extension='true'):

        allow_RGBA_output = (allow_RGBA_output == 'true')

        if not os.path.exists(path):
            return (None, )
        fl = self.BatchImageLoader(path, label, pattern)
        new_paths = fl.image_paths
        if mode == 'single_image':
            image, filename = fl.get_image_by_id(index)
            if image == None:
                cstr(f"No valid image was found for the inded `{index}`").error.print()
                return (None, None)
        elif mode == 'incremental_image':
            image, filename = fl.get_next_image()
            if image == None:
                cstr(f"No valid image was found for the next ID. Did you remove images from the source directory?").error.print()
                return (None, None)
        else:
            newindex = int(random.random() * len(fl.image_paths))
            image, filename = fl.get_image_by_id(newindex)
            if image == None:
                cstr(f"No valid image was found for the next ID. Did you remove images from the source directory?").error.print()
                return (None, None)

        # Update history
        update_history_images(new_paths)

        if not allow_RGBA_output:
           image = image.convert("RGB")

        text = self.load_txt_file(f'{path}/{os.path.splitext(filename)[0]}.txt')

        filepath = os.path.join(path, filename)

        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]

        return (pil2tensor(image), text, filename, filepath)

    class BatchImageLoader:
        def __init__(self, directory_path, label, pattern):
            self.WDB = WDB
            self.image_paths = []
            self.load_images(directory_path, pattern)
            self.image_paths.sort()
            stored_directory_path = self.WDB.get('Batch Paths', label)
            stored_pattern = self.WDB.get('Batch Patterns', label)
            if stored_directory_path != directory_path or stored_pattern != pattern:
                self.index = 0
                self.WDB.insert('Batch Counters', label, 0)
                self.WDB.insert('Batch Paths', label, directory_path)
                self.WDB.insert('Batch Patterns', label, pattern)
            else:
                self.index = self.WDB.get('Batch Counters', label)
            self.label = label

        def load_images(self, directory_path, pattern):
            for file_name in glob.glob(os.path.join(glob.escape(directory_path), pattern), recursive=True):
                if file_name.lower().endswith(ALLOWED_EXT):
                    abs_file_path = os.path.abspath(file_name)
                    self.image_paths.append(abs_file_path)

        def get_image_by_id(self, image_id):
            if image_id < 0 or image_id >= len(self.image_paths):
                cstr(f"Invalid image index `{image_id}`").error.print()
                return
            i = Image.open(self.image_paths[image_id])
            i = ImageOps.exif_transpose(i)
            return (i, os.path.basename(self.image_paths[image_id]))

        def get_next_image(self):
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]
            self.index += 1
            if self.index == len(self.image_paths):
                self.index = 0
            cstr(f'{cstr.color.YELLOW}{self.label}{cstr.color.END} Index: {self.index}').msg.print()
            self.WDB.insert('Batch Counters', self.label, self.index)
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            return (i, os.path.basename(image_path))

        def get_current_image(self):
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]
            return os.path.basename(image_path)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs['mode'] != 'single_image':
            return float("NaN")
        else:
            fl = LoadImagesPairBatchNode.BatchImageLoader(kwargs['path'], kwargs['label'], kwargs['pattern'])
            filename = fl.get_current_image()
            image = os.path.join(kwargs['path'], filename)
            sha = get_sha256(image)
            return sha

class SaveImagesPairNode:
    def __init__(self):
        self.output_dir = comfy_paths.output_directory
        self.output_path = ''
        self.type = 'output'
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "output_path": ("STRING", {"default": '[time(%Y-%m-%d)]', "multiline": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "filename_delimiter": ("STRING", {"default":"_"}),
                "filename_number_padding": ("INT", {"default":4, "min":1, "max":9, "step":1}),
                "extension": (['png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'bmp'], ),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                "lossless_webp": (["false", "true"],),
                "show_previews": (["true", "false"],),
                "try_copy_to_output": (["true", "false"],),
            },
            "optional": {
                "image": ("IMAGE", ),
                "image_path": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image_pair"

    OUTPUT_NODE = True

    CATEGORY = "Aimidi Nodes"

    def save_text_file(self, text, output_path, filename_prefix, filename_delimiter, filename_number_padding):

        tokens = TextTokens()
        filename_prefix = tokens.parseTokens(filename_prefix)

        # Setup output path
        if output_path in [None, '', "none", "."]:
            output_path = self.output_dir
        else:
            output_path = tokens.parseTokens(output_path)
        if not os.path.isabs(output_path):
            output_path = os.path.join(self.output_dir, output_path)

        # Check output destination
        if output_path.strip() != '':
            if not os.path.isabs(output_path):
                output_path = os.path.join(comfy_paths.output_directory, output_path)
            if not os.path.exists(output_path.strip()):
                cstr(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.').warning.print()
                os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path

        if text.strip() == '':
            cstr(f"There is no text specified to save! Text is empty.").error.print()

        delimiter = filename_delimiter
        number_padding = int(filename_number_padding)
        file_extension = '.txt'
        filename = self.generate_filename(output_path, filename_prefix, delimiter, number_padding, file_extension)
        file_path = os.path.join(output_path, filename)

        self.writeTextFile(file_path, text)

        update_history_text_files(file_path)

        return os.path.splitext(filename)[0]

    def generate_filename(self, path, prefix, delimiter, number_padding, extension):
        pattern = f"{re.escape(prefix)}{re.escape(delimiter)}(\\d{{{number_padding}}})"
        existing_counters = [
            int(re.search(pattern, filename).group(1))
            for filename in os.listdir(path)
            if re.match(pattern, filename)
        ]
        existing_counters.sort(reverse=True)

        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        filename = f"{prefix}{delimiter}{counter:0{number_padding}}{extension}"
        while os.path.exists(os.path.join(path, filename)):
            counter += 1
            filename = f"{prefix}{delimiter}{counter:0{number_padding}}{extension}"

        return filename

    def writeTextFile(self, file, content):
        try:
            with open(file, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
        except OSError:
            cstr(f"Unable to save file `{file}`").error.print()  

    def was_save_images(self, images, filename, extension, quality, lossless_webp, show_previews, try_copy_to_output, image_path=''):
        
        lossless_webp = (lossless_webp == "true")

        # Set Extension
        file_extension = '.' + extension
        if file_extension not in ALLOWED_EXT:
            cstr(f"The extension `{extension}` is not valid. The valid formats are: {', '.join(sorted(ALLOWED_EXT))}").error.print()
            file_extension = "png"

        # Delegate the filename stuffs
        file = f"{filename}{file_extension}"
        output_file = os.path.abspath(os.path.join(self.output_path, file))

        results = list()
        if images is not None:
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # Delegate metadata/pnginfo
                if extension == 'webp':
                    img_exif = img.getexif()
                    exif_data = img_exif.tobytes()
                else:
                    metadata = PngInfo()
                    exif_data = metadata

                if not try_copy_to_output or image_path == '':
                    # Save the images
                    try:
                        if extension in ["jpg", "jpeg"]:
                            img.save(output_file,
                                    quality=quality, optimize=True)
                        elif extension == 'webp':
                            img.save(output_file,
                                    quality=quality, lossless=lossless_webp, exif=exif_data)
                        elif extension == 'png':
                            img.save(output_file,
                                    pnginfo=exif_data, optimize=True)
                        elif extension == 'bmp':
                            img.save(output_file)
                        elif extension == 'tiff':
                            img.save(output_file,
                                    quality=quality, optimize=True)
                        else:
                            img.save(output_file,
                                    pnginfo=exif_data, optimize=True)

                        cstr(f"Image file saved to: {output_file}").msg.print()

                        if show_previews == 'true':
                            subfolder = self.get_subfolder_path(output_file, self.output_dir)
                            results.append({
                                "filename": file,
                                "subfolder": subfolder,
                                "type": self.type
                            })

                        # Update the output image history
                        update_history_output_images(output_file)

                    except OSError as e:
                        cstr(f'Unable to save file to: {output_file}').error.print()
                        print(e)
                    except Exception as e:
                        cstr('Unable to save file due to the to the following error:').error.print()
                        print(e)
                else:
                    # Copy the image
                    self.copy_images(show_previews, image_path, output_file, filename, results)
        elif image_path != '':
            # Copy the image
                self.copy_images(show_previews, image_path, output_file, filename, results)

        if show_previews == 'true':
            return results
        else:
            return []

    def copy_images(self, show_previews, src_image_path, output_file, filename, results):
        try:
            import shutil
            # 获取原文件的扩展名
            src_extension = os.path.splitext(src_image_path)[1]
            # 设置目标文件的扩展名
            output_file = os.path.splitext(output_file)[0] + src_extension

            shutil.copy2(src_image_path, output_file)

            cstr(f"Image file copied to: {output_file}").msg.print()

            if show_previews == 'true':
                subfolder = self.get_subfolder_path(output_file, self.output_dir)
                results.append({
                                "filename": filename + src_extension,
                                "subfolder": subfolder,
                                "type": self.type
                            })

                        # Update the output image history
            update_history_output_images(output_file)

        except OSError as e:
            cstr(f'Unable to copy file to: {output_file}').error.print()
            print(e)
        except Exception as e:
            cstr('Unable to copy file due to the to the following error:').error.print()
            print(e)

    def get_subfolder_path(self, image_path, output_path):
        output_parts = output_path.strip(os.sep).split(os.sep)
        image_parts = image_path.strip(os.sep).split(os.sep)
        common_parts = os.path.commonprefix([output_parts, image_parts])
        subfolder_parts = image_parts[len(common_parts):]
        subfolder_path = os.sep.join(subfolder_parts[:-1])
        return subfolder_path

    def save_image_pair(self, text, output_path='', filename_prefix="ComfyUI", filename_delimiter='_',
                        extension='png', quality=100, lossless_webp="false", filename_number_padding=4, show_previews="true", 
                        try_copy_to_output='true', image_path='', image = None):
        filename = self.save_text_file(text, output_path, filename_prefix, filename_delimiter, filename_number_padding)
        previews = self.was_save_images(image, filename, extension, quality, lossless_webp, show_previews, try_copy_to_output, image_path)
        return {"ui": {"images": previews}}