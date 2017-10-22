"""Global registry for feature handlers."""



def register_feature_handler(feature_handler_name):
    def decorator(feature_handler):
        global FEATURE_HANDLER_MAP
        assert feature_handler_name not in FEATURE_HANDLER_MAP
        FEATURE_HANDLER_MAP[feature_handler_name] = feature_handler
        return feature_handler
    return decorator


def get_feature_handler(feature_handler_name):
    """Get function to retrieve features."""
    global FEATURE_HANDLER_MAP
    return FEATURE_HANDLER_MAP[feature_handler_name]


FEATURE_HANDLER_MAP = {}
