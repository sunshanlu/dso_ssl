# pelicanconf.py

# 网站作者的名称
AUTHOR = "ssl"

# 网站的名称或标题
SITENAME = "SSL的SLAM系列之DSO"

# 网站的根 URL。在开发阶段通常留空，但在生产环境中应设置为实际的域名
SITEURL = ""

# 包含内容文件（如 Markdown 文件）的目录路径
PATH = "content"

# 网站的时间区域设置。用于正确显示日期和时间
TIMEZONE = "Asia/Shanghai"  # 建议使用标准的时区名称

# 网站的默认语言代码
DEFAULT_LANG = "zh-cn"

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# 用于网站和其他资源的链接
LINKS = (("Pelican", "https://getpelican.com/"), ("Python.org", "https://www.python.org/"), ("Jinja2", "https://palletsprojects.com/p/jinja/"))

# 用于填写社交主页的链接
SOCIAL = (
    ("BiliBili", "https://space.bilibili.com/489032586"),
    ("GitHub", "https://github.com/sunshanlu"),
)

# 每页显示的文章数量。设置为 False 表示不分页
DEFAULT_PAGINATION = False

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True

# 指定使用的主题名称或路径
THEME = "elegant"

# 配置插件信息
PLUGIN_PATHS = ["plugins"]
# PLUGINS = ["merge_plugin", "pelican.plugins.render_math", "pelican.plugins.search"]
PLUGINS = ["pelican.plugins.render_math", "pelican.plugins.search"]

# ======================= elegant 主题配置 =======================

# 启用图标
USE_SHORTCUT_ICONS = True
STATIC_PATHS = ["theme/images", "images"]

# 搜索配置
STORK_INPUT_OPTIONS = {"base_directory": PATH}
