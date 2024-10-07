import configparser


def read_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def get_config_section(config, section):
    if config.has_section(section):
        return dict(config.items(section))
    else:
        raise Exception(f"Section {section} not found in the config file")


# Example usage:
# config = read_config('path/to/config.ini')
# db_config = get_config_section(config, 'database')
