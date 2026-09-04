"""Mint a Copernicus Data Space (CDSE) S3 key pair from the terminal and write it
to .env, without the website. The recipe is CDSE's own (documentation.dataspace.
copernicus.eu/APIs/S3.html, "Example script to download product using python"):

  1. a Keycloak access token by password grant (client_id cdse-public)
  2. POST https://s3-keys-manager.cloudferro.com/api/user/credentials
     with that token -> {"access_id": ..., "secret": ...}

Usage:  uv run python cdse_keys.py            (prompts for username and password)
        uv run python cdse_keys.py --list     (the key pairs the account holds)
        uv run python cdse_keys.py --delete ACCESS_ID

The pair is appended to .env in the repo as CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY
(the notebook loads .env with python-dotenv). The secret is shown once by the
API, so it is written straight to the file and not printed. An account with
two-factor authentication on will not pass the password grant; make the keys
on the site in that case. Standard library only.
"""
import getpass
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
KEYS_URL = "https://s3-keys-manager.cloudferro.com/api/user/credentials"
ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _call(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
            return r.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return e.code, body


def token(user, pw):
    data = urllib.parse.urlencode({
        "client_id": "cdse-public", "grant_type": "password", "username": user, "password": pw,
    }).encode()
    st, out = _call(TOKEN_URL, data, {"Content-Type": "application/x-www-form-urlencoded"})
    if st != 200:
        sys.exit(f"token: HTTP {st}: {str(out)[:300]}")
    return out["access_token"]


def main():
    args = sys.argv[1:]
    user = os.environ.get("CDSE_USERNAME") or input("CDSE username (email): ").strip()
    pw = os.environ.get("CDSE_PASSWORD") or getpass.getpass("CDSE password: ")
    hdr = {"Authorization": "Bearer " + token(user, pw)}
    if args[:1] == ["--list"]:
        st, out = _call(KEYS_URL, headers=hdr)
        print(st, json.dumps(out, indent=2) if not isinstance(out, str) else out[:500])
        return
    if args[:1] == ["--delete"] and len(args) > 1:
        st, out = _call(f"{KEYS_URL}/access_id/{args[1]}", headers=hdr, method="DELETE")
        print(st, out if isinstance(out, str) else json.dumps(out))
        return
    st, out = _call(KEYS_URL, headers=hdr, method="POST")
    if st != 200 or not isinstance(out, dict) or "access_id" not in out:
        sys.exit(f"keys manager: HTTP {st}: {str(out)[:500]}")
    with open(ENV, "a") as f:
        f.write(f"\nCDSE_S3_ACCESS_KEY={out['access_id']}\nCDSE_S3_SECRET_KEY={out['secret']}\n")
    print(f"access id {out['access_id']} written with its secret to {ENV}")
    print("restart the notebook kernel so it reads the pair")


if __name__ == "__main__":
    main()
