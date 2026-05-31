"""Kubera <-> IBKR integration.

Auto-detects whichever account is logged into the running IB Gateway / TWS
session (paper or live) by probing standard ports and reading the account
number prefix. The user changes modes by changing what they log into in
Gateway, not by changing any config or env var here.
"""
