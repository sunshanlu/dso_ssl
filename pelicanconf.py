import os

AUTHOR = "孙善路"
SITENAME = "SSL的SLAM系列 - DSO"
SITESUBTITLE = "最好的DSO解析系列"
SITEURL = ""

PATH = "content"

# Regional Settings
TIMEZONE = "Asia/Shanghai"
DATE_FORMATS = {"en": "%Y-%m-%d"}

DEFAULT_LANG = "en"

# Plugins and extensions
MARKDOWN = {
    "extension_configs": {
        "markdown.extensions.admonition": {},
        "markdown.extensions.codehilite": {"css_class": "highlight"},
        "markdown.extensions.extra": {},
        "markdown.extensions.meta": {},
        "markdown.extensions.toc": {},
    }
}

PLUGIN_PATHS = ["plugins"]
PLUGINS = [
    "siteurl_replacer",
    "pelican.plugins.extract_toc",
    "pelican.plugins.tipue_search",
    "pelican.plugins.liquid_tags.img",
    "pelican.plugins.liquid_tags.include_code",
    "pelican.plugins.neighbors",
    "pelican.plugins.related_posts",
    "pelican.plugins.series",
    "pelican.plugins.share_post",
    "pelican.plugins.sitemap",
    "pelican.plugins.render_math",
]

MATH_JAX = {"color": "#145402", "align": "center"}

# 用于配置站点地图，方便搜索引擎的内容抓取
SITEMAP = {
    "format": "xml",
    "priorities": {"articles": 0.5, "indexes": 0.5, "pages": 0.5},
    "changefreqs": {"articles": "monthly", "indexes": "daily", "pages": "monthly"},
}

# Appearance
THEME = "elegant"
TYPOGRIFY = True
DEFAULT_PAGINATION = False

# Defaults
DEFAULT_CATEGORY = "DSO"
USE_FOLDER_AS_CATEGORY = False
ARTICLE_URL = "{slug}"
PAGE_URL = "{slug}"
PAGE_SAVE_AS = "{slug}.html"
TAGS_URL = "tags"
CATEGORIES_URL = "categories"
ARCHIVES_URL = "archives"
SEARCH_URL = "search"

# Feeds
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None
CATEGORY_FEED_ATOM = None
CATEGORY_FEED_RSS = None

# Social
SOCIAL = (
    ("Github", "https://github.com/sunshanlu", "Github 页面"),
    ("Email", "ssl2001@126.com", "Email Me"),
    ("RSS", SITEURL + "/feeds/all.atom.xml"),
)

# Elegant Themes
STATIC_PATHS = ["theme/images", "images", "files"]
EXTRA_PATH_METADATA = dict()

STATIC_PATHS.append("extra/robots.txt")
EXTRA_PATH_METADATA["extra/robots.txt"] = {"path": "robots.txt"}

STATIC_PATHS.append("theme/images/favicon.ico")
EXTRA_PATH_METADATA["theme/images/favicon.ico"] = {"path": "favicon.ico"}

DIRECT_TEMPLATES = ["index", "tags", "categories", "archives", "search", "404"]
TAG_SAVE_AS = ""
AUTHOR_SAVE_AS = ""
CATEGORY_SAVE_AS = ""
USE_SHORTCUT_ICONS = True

# Elegant Labels
SOCIAL_PROFILE_LABEL = "Contact Me"
RELATED_POSTS_LABEL = "Keep Reading"
SHARE_POST_INTRO = "喜欢这篇文章吗？喜欢就分享吧:"
COMMENTS_INTRO = "有什么问题吗，有任何问题欢迎你在下面评论留言或者邮件联系我！"

# Legal
SITE_LICENSE = """Content licensed under <a rel="license nofollow noopener noreferrer"
    href="http://creativecommons.org/licenses/by/4.0/" target="_blank">
    Creative Commons Attribution 4.0 International License</a>."""
HOSTED_ON = {"name": "Github", "url": "https://www.github.com/"}

# Search Engine Optimization
SITE_DESCRIPTION = "SSL的SLAM系列 - DSO解析"

# Share links at bottom of articles
# Supported: twitter, facebook, hacker-news, reddit, linkedin, email
SHARE_LINKS = [("twitter", "Twitter"), ("facebook", "Facebook"), ("email", "Email")]

# Landing Page
PROJECTS_TITLE = "相关项目"
PROJECTS = [
    {
        "name": "Pelican",
        "url": "https://getpelican.com/",
        "description": "网站由Pelican框架生成。",
    },
    {
        "name": "Elegant",
        "url": "https://github.com/Pelican-Elegant/elegant",
        "description": "网站使用的Pelican主题。",
    },
    {
        "name": "DSO_SSL",
        "url": "https://github.com/sunshanlu/dso_ssl",
        "description": "SSL重写DSO的仓库地址。",
    },
    {
        "name": "Github-SSL",
        "url": "https://github.com/sunshanlu",
        "description": "SSL的Github主页，看看是否有你感兴趣的项目呢？",
    },
]

LANDING_PAGE_TITLE = "DSO – vSLAM直接法的神，SSL的DSO解析"

AUTHORS = {
    "孙善路-github": {
        "url": "https://github.com/sunshanlu",
        "blurb": "对SLAM和DL感兴趣的理工男",
        "avatar": "https://avatars.githubusercontent.com/u/78467062",
    },
    "孙善路-bilibili": {
        "url": "https://space.bilibili.com/489032586",
        "blurb": "对SLAM和DL感兴趣的理工男",
        "avatar": SITEURL + "/images/avatars/sunshanlu.png",
    },
}


# UTTERANCES配置
UTTERANCES_REPO = "sunshanlu/dso_ssl"
UTTERANCES_THEME = "github-light"
UTTERANCES_LABEL = "✨💬✨"

# SEO配置
CLAIM_BING = "A86784EE624E2187E0AC969B764CE1E1"
CLAIM_GOOGLE = "UpDPegqFb_TnOIJbDp5ud8wVncaUoUB-fJoQyiiAdmg"

# 启用 Jinja2 支持
JINJA_ENVIRONMENT = {
    "extensions": [],
    "trim_blocks": True,
    "lstrip_blocks": True,
}
