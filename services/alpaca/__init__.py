"""Alpaca trading integration for Kubera.

Mirrors the services/ibkr/ shape — connection, sync loop, order engine,
reconciliation. Swap between paper and live via ALPACA_MODE env var
(or pass mode= explicitly when constructing KuberaAlpaca).
"""
