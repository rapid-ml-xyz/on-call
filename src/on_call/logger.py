import logging

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
