import yaml
from typing import Any, Dict, Optional


class ConfigManager:
    """
    Loads config from a file or dictionary.
    Provides access to sections needed by other modules.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        self._config = {}
        if config_path:
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)
        if extra_params:
            # Merge or override config with extra_params
            self._config.update(extra_params)

    def get_config_section(self, section_name: str) -> Dict[str, Any]:
        return self._config.get(section_name, {})
