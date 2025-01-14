# siteurl_replacer.py

import os
import shutil
import re
from pelican import signals

# 存储原始文件内容的字典
original_contents = {}


def replace_siteurl(generator):
    """
    Replace {{ SITEURL }} in Markdown files with the actual SITEURL from pelicanconf.py
    """
    siteurl = generator.settings.get("SITEURL", "")

    for root, _, files in os.walk(generator.settings["PATH"]):
        for file in files:
            if file.endswith((".md", ".markdown", ".mkd", ".mdown", ".mdwn", ".mkdn", ".mdtxt", ".mdtext")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                original_contents[file_path] = content  # 存储原始内容
                modified_content = content.replace("{{ SITEURL }}", siteurl)
                modified_content = re.sub(r'(?<!\w)(\.\.?/)', siteurl + '/', modified_content)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(modified_content)


def restore_siteurl(generator):
    """
    Restore the original content of Markdown files after generation
    """
    for file_path, original_content in original_contents.items():
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(original_content)


def register():
    """
    Register the plugin with Pelican
    """
    signals.initialized.connect(replace_siteurl)
    signals.finalized.connect(restore_siteurl)
