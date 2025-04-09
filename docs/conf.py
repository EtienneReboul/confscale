extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]
source_suffix = ".rst"
master_doc = "index"
project = "confscale"
year = "2025"
author = "Etienne Reboul"
copyright = f"{year}, {author}"
try:
    from pkg_resources import get_distribution

    version = release = get_distribution("confscale").version
except Exception:
    import traceback

    traceback.print_exc()
    version = release = "0.0.0"

pygments_style = "trac"
templates_path = ["."]
extlinks = {
    "issue": ("https://github.com/EtienneReboul/confscale/issues/%s", "#%s"),
    "pr": ("https://github.com/EtienneReboul/confscale/pull/%s", "PR #%s"),
}

html_theme = "furo"
html_theme_options = {
    "githuburl": "https://github.com/EtienneReboul/confscale/",
}

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_short_title = f"{project}-{version}"

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
