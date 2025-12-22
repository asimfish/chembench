"""Package containing the core framework."""

import os
import toml

# Conveniences to other module directories via relative paths
PSILAB_TEXTURE_ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets/texture"))
PSILAB_USD_ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets/usd"))
PSILAB_URDF_ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets/urdf"))

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../outputs"))
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../logs"))
SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts_psi"))

# Conveniences to other module directories via relative paths
PSILAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

PSILAB_METADATA = toml.load(os.path.join(PSILAB_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = PSILAB_METADATA["package"]["version"]