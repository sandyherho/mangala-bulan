from pathlib import Path

class ConfigManager:
    @staticmethod
    def load(config_path: str) -> dict:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        config = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                config[key] = ConfigManager._parse_value(value)
        return config
    
    @staticmethod
    def _parse_value(value: str):
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        try:
            # Try to evaluate mathematical expressions
            if '/' in value or '*' in value or '+' in value or '-' in value:
                try:
                    return eval(value)
                except:
                    pass
            # Standard parsing
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
