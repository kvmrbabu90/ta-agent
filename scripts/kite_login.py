"""Interactive helper to mint a fresh Kite Connect access token.

Kite tokens expire daily at ~6am IST. The login flow is necessarily
interactive (Zerodha requires manual browser auth), so this script:

    1. Prints the login URL to open in a browser.
    2. After redirect, asks you to paste the ``request_token`` query parameter.
    3. Exchanges it for an ``access_token``.
    4. Prints the line to add to your ``.env``.

Run:
    python -m scripts.kite_login
"""

from __future__ import annotations

import click
from kiteconnect import KiteConnect

from packages.common.config import settings


@click.command()
def main() -> None:
    if not settings.kite_api_key or not settings.kite_api_secret:
        raise click.UsageError(
            "KITE_API_KEY and KITE_API_SECRET must be set in .env first."
        )

    kite = KiteConnect(api_key=settings.kite_api_key)

    click.echo("=" * 70)
    click.echo("Kite Connect login")
    click.echo("=" * 70)
    click.echo("\n1. Open this URL in your browser and log in:\n")
    click.echo(f"   {kite.login_url()}\n")
    click.echo("2. After successful login Zerodha redirects to your registered")
    click.echo("   redirect URL. Look in the resulting URL for a query param")
    click.echo("   named 'request_token=...' and copy its value.\n")

    request_token = click.prompt("3. Paste the request_token", type=str).strip()

    try:
        session = kite.generate_session(
            request_token, api_secret=settings.kite_api_secret
        )
    except Exception as exc:  # noqa: BLE001 — surface the SDK error verbatim
        raise click.ClickException(f"generate_session failed: {exc!r}") from exc

    access_token = session.get("access_token")
    if not access_token:
        raise click.ClickException(
            f"No access_token in response: {session!r}"
        )

    click.echo("\nSuccess. Add this line to your .env (replacing any existing):\n")
    click.echo(f"   KITE_ACCESS_TOKEN={access_token}")
    click.echo(
        "\nThis token is valid until ~6am IST tomorrow. Re-run this script "
        "to refresh."
    )


if __name__ == "__main__":
    main()
