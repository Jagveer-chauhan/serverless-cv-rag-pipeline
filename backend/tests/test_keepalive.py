"""Tests for the Keepalive cron webhook and health status endpoints."""
import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "sla_target_ms" in data
        assert "X-Process-Time-Ms" in response.headers


@pytest.mark.asyncio
async def test_keepalive_get():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/keepalive")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0
        assert "memory_usage_mb" in data
        assert data["memory_usage_mb"] > 0
        assert data["sla_target_ms"] == 5000.0
        assert "timestamp" in data
        assert "X-Process-Time-Ms" in response.headers


@pytest.mark.asyncio
async def test_keepalive_post():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/keepalive")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "Render worker is active" in data["message"]
