"""Pytest configuration for the PokerVision backend.

Makes the backend package root importable so tests can do
``from ml.hand_evaluator import ...``, ``from services...`` and
``from main import app`` whether pytest is invoked as ``pytest`` or
``python -m pytest`` from the ``backend/`` directory.
"""
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
