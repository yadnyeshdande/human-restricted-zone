# =============================================================================
# File: config/migration.py
# =============================================================================
"""Configuration migration and backward compatibility."""

from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger("Migration")


def migrate_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate old config formats to current version."""
    version = data.get('app_version', '0.0.0')
    
    if version == '1.0.0':
        return data
    
    # Add migration logic here for future versions
    logger.info(f"Migrated config from version {version} to 1.0.0")
    data['app_version'] = '1.0.0'
    return data
