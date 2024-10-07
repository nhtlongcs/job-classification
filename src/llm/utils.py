def switch_api_key(model, API_KEYS, exhausted_key=None):

    if exhausted_key:
        API_KEYS.remove(exhausted_key)
        API_KEYS.append(exhausted_key)

    if not API_KEYS:
        raise ValueError("All API keys have been exhausted")

    new_key = API_KEYS[0]
    model.configure(api_key=new_key)

    print(f"Switched to a new API key: {new_key[:10]}...")

    return new_key, API_KEYS
