import pytest
from dotenv import load_dotenv
import os

@pytest.fixture(scope="session", autouse=True)
def load_env():
    # Automatically load .env at the beginning of the test session
    load_dotenv()