# This file is only used if you use `make publish` or
# explicitly specify it as your config file.

import os
import sys

sys.path.append(os.curdir)
from pelicanconf import *


SITEURL = "https://sunshanlu.github.io/dso_ssl"
RELATIVE_URLS = False

# 主要是方便RSS阅读器和搜索引擎的检索
FEED_ALL_ATOM = "feeds/all.atom.xml"            # 生成所有文章的 Atom 订阅源文件
CATEGORY_FEED_ATOM = "feeds/{slug}.atom.xml"    # 生成每个分类的 Atom 订阅源文件

# 在构建前删除输出目录
DELETE_OUTPUT_DIRECTORY = True

# 以下项目在发布时通常很有用
# DISQUS_SITENAME = ""      # 评论系统相关
# GOOGLE_ANALYTICS = ""     # 谷歌网站评估相关
