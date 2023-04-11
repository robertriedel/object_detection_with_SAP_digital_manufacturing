import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session


def get_access_token(token_endpoint, client_id, client_secret):
    # Create a client object with the client credentials
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)

    # Fetch the access token using the client credentials
    token = oauth.fetch_token(
        token_url=token_endpoint,
        client_id=client_id,
        client_secret=client_secret
    )

    # Get the access token from the token response
    access_token = token["access_token"]
    return access_token
