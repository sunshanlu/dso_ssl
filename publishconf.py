import os
import sys

sys.path.append(os.curdir)
from pelicanconf import *


SITEURL = "https://sunshanlu.github.io/dso_ssl"
FEED_DOMAIN = SITEURL
FEED_ALL_ATOM = "feeds/all.atom.xml"
CATEGORY_FEED_ATOM = "feeds/{slug}.atom.xml"
STAT_COUNTER_PROJECT = os.environ.get("STAT_COUNTER_PROJECT_PROD")
STAT_COUNTER_SECURITY = os.environ.get("STAT_COUNTER_SECURITY_PROD")
GOOGLE_ANALYTICS = os.environ.get("GOOGLE_ANALYTICS_PROD")

RELATIVE_URLS = False
DELETE_OUTPUT_DIRECTORY = True

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
