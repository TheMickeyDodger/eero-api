"""Tests for NetworksAPI module.

Tests cover:
- Getting network list / a single network (raw response)
- Rebooting the network and running/reading speed tests (empty-string body,
  query-parameter reads)
- Renaming the network and setting/clearing its password (form-encoded
  writes, client-disconnecting)
- Guest network reads/writes and its own password sub-resource
- id/path/URL polymorphism and parent-link preference
- The uncharacterised-write warning on every unverified write
- Rejecting non-string and empty Wi-Fi passwords before any request
- Rejecting non-string and empty network / supplied guest names before any request
- That no method mutates its ``parent`` argument
"""

import contextlib
import copy
import logging
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qsl

import aiohttp
import pytest

from eero.api.networks import NetworksAPI
from eero.exceptions import EeroAuthenticationException, EeroValidationException

from .conftest import (
    INVALID_GUEST_NAMES,
    INVALID_NETWORK_NAMES,
    INVALID_PASSWORDS,
    PASSWORD_WRITES,
    VALID_NAMES,
    VALID_PASSWORDS,
    api_success_response,
    create_mock_response,
)


@pytest.fixture
def networks_api(mock_session):
    """Create a NetworksAPI with mocked auth."""
    auth_api = MagicMock()
    auth_api.session = mock_session
    auth_api.get_auth_token = AsyncMock(return_value="auth_token")
    return NetworksAPI(auth_api)


def _form_fields_sent(mock_session) -> List[Dict[str, str]]:
    """Decode each request's ``data=`` mapping as aiohttp would put it on the wire."""
    return [
        dict(parse_qsl(aiohttp.FormData(call.kwargs["data"])().decode(), keep_blank_values=True))
        for call in mock_session.request.call_args_list
    ]


class TestNetworksAPIInit:
    """Tests for NetworksAPI initialization."""

    def test_init_with_auth_api(self, mock_session):
        """Test initialization with AuthAPI."""
        auth_api = MagicMock()
        auth_api.session = mock_session

        api = NetworksAPI(auth_api)

        assert api._auth_api is auth_api


class TestNetworksAPIGetNetworks:
    """Tests for get_networks method."""

    @pytest.mark.asyncio
    async def test_get_networks_returns_raw_response(
        self, networks_api, mock_session, sample_networks_list
    ):
        """Test get_networks returns raw API response."""
        expected_response = api_success_response({"networks": sample_networks_list})
        mock_session.request.return_value = create_mock_response(200, expected_response)

        result = await networks_api.get_networks()

        assert "meta" in result
        assert result["data"]["networks"] == sample_networks_list

    @pytest.mark.asyncio
    async def test_get_networks_not_authenticated(self, networks_api):
        """Test get_networks raises when not authenticated."""
        networks_api._auth_api.get_auth_token = AsyncMock(return_value=None)

        with pytest.raises(EeroAuthenticationException, match="Not authenticated"):
            await networks_api.get_networks()


class TestNetworksAPIGetNetwork:
    """Tests for get_network method."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "network_id,expected_suffix",
        [
            ("network_123", "/2.2/networks/network_123"),
            ("/2.2/networks/network_123", "/2.2/networks/network_123"),
            ("https://api-user.e2ro.com/2.2/networks/network_123", "/2.2/networks/network_123"),
        ],
    )
    async def test_get_network_accepts_id_path_or_url(
        self, networks_api, mock_session, sample_network_data, network_id, expected_suffix
    ):
        """Test get_network accepts a bare id, a path, or an absolute URL."""
        expected_response = api_success_response(sample_network_data)
        mock_session.request.return_value = create_mock_response(200, expected_response)

        result = await networks_api.get_network(network_id)

        assert result["data"]["id"] == "network_123"
        call_args = mock_session.request.call_args
        assert call_args.args[1].endswith(expected_suffix)

    @pytest.mark.asyncio
    async def test_get_network_prefers_parent_self_url(
        self, networks_api, mock_session, sample_network_data
    ):
        """Test get_network uses the parent's own url over the template."""
        expected_response = api_success_response(sample_network_data)
        mock_session.request.return_value = create_mock_response(200, expected_response)
        parent = {"url": "/2.3/networks/network_123"}

        await networks_api.get_network("network_123", parent=parent)

        call_args = mock_session.request.call_args
        assert call_args.args[1].endswith("/2.3/networks/network_123")

    @pytest.mark.asyncio
    async def test_get_network_not_authenticated(self, networks_api):
        """Test get_network raises when not authenticated."""
        networks_api._auth_api.get_auth_token = AsyncMock(return_value=None)

        with pytest.raises(EeroAuthenticationException):
            await networks_api.get_network("network_123")


class TestNetworksAPIReboot:
    """Tests for network reboot functionality."""

    @pytest.mark.asyncio
    async def test_reboot_network_posts_empty_json_string(self, networks_api, mock_session):
        """Test reboot_network POSTs the two-byte "" body to the reboot link."""
        mock_session.request.return_value = create_mock_response(
            200, {"meta": {"code": 200}, "data": {}}
        )

        result = await networks_api.reboot_network("network_123")

        assert "meta" in result
        call_args = mock_session.request.call_args
        assert call_args.args[0] == "POST"
        assert call_args.args[1].endswith("/2.2/networks/network_123/reboot")
        assert call_args.kwargs["data"] == '""'

    @pytest.mark.asyncio
    async def test_reboot_network_prefers_parent_link(self, networks_api, mock_session):
        """Test reboot_network uses the network's published reboot link."""
        mock_session.request.return_value = create_mock_response(
            200, {"meta": {"code": 200}, "data": {}}
        )
        parent = {"resources": {"reboot": "/2.3/networks/network_123/reboot"}}
        parent_copy = copy.deepcopy(parent)

        await networks_api.reboot_network("network_123", parent=parent)

        call_args = mock_session.request.call_args
        assert call_args.args[1].endswith("/2.3/networks/network_123/reboot")
        assert parent == parent_copy

    @pytest.mark.asyncio
    async def test_reboot_network_not_authenticated(self, networks_api):
        """Test reboot raises when not authenticated."""
        networks_api._auth_api.get_auth_token = AsyncMock(return_value=None)

        with pytest.raises(EeroAuthenticationException):
            await networks_api.reboot_network("network_123")

    @pytest.mark.asyncio
    async def test_reboot_network_warns_uncharacterised_write(
        self, networks_api, mock_session, caplog
    ):
        """Test reboot_network logs the uncharacterised-write warning once."""
        mock_session.request.return_value = create_mock_response(
            200, {"meta": {"code": 200}, "data": {}}
        )

        with caplog.at_level(logging.WARNING):
            await networks_api.reboot_network("network_123")

        assert any("reboot network" in m for m in caplog.messages)
        assert not any("network_123" in m for m in caplog.messages)


class TestNetworksAPISpeedTest:
    """Tests for speed test functionality."""

    @pytest.mark.asyncio
    async def test_run_speed_test_posts_empty_json_string(self, networks_api, mock_session):
        """Test run_speed_test POSTs the two-byte "" body to the speedtest link."""
        expected_response = api_success_response({"down": {"value": 500.0}})
        mock_session.request.return_value = create_mock_response(200, expected_response)

        result = await networks_api.run_speed_test("network_123")

        assert result["data"]["down"]["value"] == 500.0
        call_args = mock_session.request.call_args
        assert call_args.args[0] == "POST"
        assert call_args.args[1].endswith("/2.2/networks/network_123/speedtest")
        assert call_args.kwargs["data"] == '""'

    @pytest.mark.asyncio
    async def test_run_speed_test_does_not_warn(self, networks_api, mock_session, caplog):
        """Test run_speed_test does not carry the uncharacterised-write warning.

        The write is live-verified, so it must not carry the warning.
        """
        mock_session.request.return_value = create_mock_response(
            200, api_success_response({"down": {"value": 500.0}})
        )

        with caplog.at_level(logging.WARNING):
            await networks_api.run_speed_test("network_123")

        assert not any("not been fully characterised" in m for m in caplog.messages)

    @pytest.mark.asyncio
    async def test_get_speed_tests_sends_query_params(self, networks_api, mock_session):
        """Test get_speed_tests GETs the speedtest link with limit/startTime/endTime."""
        mock_session.request.return_value = create_mock_response(200, api_success_response([]))

        await networks_api.get_speed_tests(
            "network_123",
            limit=5,
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-02T00:00:00Z",
        )

        call_args = mock_session.request.call_args
        assert call_args.args[0] == "GET"
        assert call_args.args[1].endswith("/2.2/networks/network_123/speedtest")
        assert call_args.kwargs["params"] == {
            "limit": "5",
            "startTime": "2026-01-01T00:00:00Z",
            "endTime": "2026-01-02T00:00:00Z",
        }

    @pytest.mark.asyncio
    async def test_get_speed_tests_omits_unsupplied_params(self, networks_api, mock_session):
        """Test get_speed_tests sends no query params when none are supplied."""
        mock_session.request.return_value = create_mock_response(200, api_success_response([]))

        await networks_api.get_speed_tests("network_123")

        call_args = mock_session.request.call_args
        assert call_args.kwargs["params"] == {}


class TestNetworksAPIPremium:
    """Tests for premium status checking."""

    @pytest.mark.asyncio
    async def test_get_premium_status_returns_raw_response(self, networks_api, mock_session):
        """Test getting premium status returns raw response."""
        network_data = {"id": "network_123", "premium_status": {"active": True}}
        mock_session.request.return_value = create_mock_response(
            200, api_success_response(network_data)
        )

        result = await networks_api.get_premium_status("network_123")

        assert result["data"]["premium_status"]["active"] is True


class TestNetworksAPISetName:
    """Tests for renaming the network."""

    @pytest.mark.asyncio
    async def test_set_network_name_sends_form_encoded_payload(
        self, networks_api, mock_session, caplog
    ):
        """Test set_network_name PUTs a form-encoded name to the settings link and warns."""
        mock_session.request.return_value = create_mock_response(
            200, {"meta": {"code": 200}, "data": {}}
        )

        with caplog.at_level(logging.WARNING):
            result = await networks_api.set_network_name("network_123", "My Network")

        assert "meta" in result
        call_args = mock_session.request.call_args
        assert call_args.args[0] == "PUT"
        assert call_args.args[1].endswith("/2.2/networks/network_123/settings")
        assert call_args.kwargs["data"] == {"name": "My Network"}
        assert any("set network name for network" in m for m in caplog.messages)


class TestNetworksAPIPassword:
    """Tests for the network's own Wi-Fi password."""

    @pytest.mark.asyncio
    async def test_set_network_password_sends_form_encoded_payload(
        self, networks_api, mock_session, caplog
    ):
        """Test set_network_password PUTs a form-encoded password and warns."""
        mock_session.request.return_value = create_mock_response(
            200, {"meta": {"code": 200}, "data": {}}
        )

        with caplog.at_level(logging.WARNING):
            await networks_api.set_network_password("network_123", "placeholder-password")

        call_args = mock_session.request.call_args
        assert call_args.args[0] == "PUT"
        assert call_args.args[1].endswith("/2.2/networks/network_123/password")
        assert call_args.kwargs["data"] == {"password": "placeholder-password"}
        assert any("set network password for network" in m for m in caplog.messages)

    @pytest.mark.asyncio
    async def test_clear_network_password_deletes_password_link(
        self, networks_api, mock_session, caplog
    ):
        """Test clear_network_password DELETEs the password link and warns."""
        mock_session.request.return_value = create_mock_response(
            200, {"meta": {"code": 200}, "data": {}}
        )

        with caplog.at_level(logging.WARNING):
            await networks_api.clear_network_password("network_123")

        call_args = mock_session.request.call_args
        assert call_args.args[0] == "DELETE"
        assert call_args.args[1].endswith("/2.2/networks/network_123/password")
        assert any("clear network password for network" in m for m in caplog.messages)


class TestNetworksAPIGuestNetwork:
    """Tests for guest network management."""

    @pytest.mark.asyncio
    async def test_get_guest_network_returns_raw_response(self, networks_api, mock_session):
        """Test get_guest_network GETs the guestnetwork link."""
        mock_session.request.return_value = create_mock_response(
            200, api_success_response({"enabled": True})
        )

        result = await networks_api.get_guest_network("network_123")

        assert result["data"]["enabled"] is True
        call_args = mock_session.request.call_args
        assert call_args.args[0] == "GET"
        assert call_args.args[1].endswith("/2.2/networks/network_123/guestnetwork")

    @pytest.mark.asyncio
    async def test_set_guest_network_sends_form_encoded_payload(
        self, networks_api, mock_session, caplog
    ):
        """Test set_guest_network PUTs form-encoded enabled/name without warning.

        The write is live-verified, so it must not carry the
        uncharacterised-write warning.
        """
        mock_session.request.return_value = create_mock_response(
            200, {"meta": {"code": 200}, "data": {}}
        )

        with caplog.at_level(logging.WARNING):
            await networks_api.set_guest_network(
                "network_123", enabled=True, name="My Guest Network"
            )

        call_args = mock_session.request.call_args
        assert call_args.args[0] == "PUT"
        assert call_args.args[1].endswith("/2.2/networks/network_123/guestnetwork")
        assert call_args.kwargs["data"] == {"enabled": "true", "name": "My Guest Network"}
        assert not any("not been fully characterised" in m for m in caplog.messages)

    @pytest.mark.asyncio
    async def test_set_guest_network_omits_name_when_not_given(self, networks_api, mock_session):
        """Test set_guest_network omits the name field when not supplied."""
        mock_session.request.return_value = create_mock_response(
            200, {"meta": {"code": 200}, "data": {}}
        )

        await networks_api.set_guest_network("network_123", enabled=False)

        call_args = mock_session.request.call_args
        assert call_args.kwargs["data"] == {"enabled": "false"}

    @pytest.mark.asyncio
    async def test_set_guest_network_prefers_parent_link(self, networks_api, mock_session):
        """Test set_guest_network uses the network's published guestnetwork link."""
        mock_session.request.return_value = create_mock_response(
            200, {"meta": {"code": 200}, "data": {}}
        )
        parent = {"resources": {"guestnetwork": "/2.3/networks/network_123/guestnetwork"}}

        await networks_api.set_guest_network("network_123", enabled=True, parent=parent)

        call_args = mock_session.request.call_args
        assert call_args.args[1].endswith("/2.3/networks/network_123/guestnetwork")


class TestNetworksAPIGuestPassword:
    """Tests for the guest network's own password sub-resource."""

    @pytest.mark.asyncio
    async def test_set_guest_password_uses_literal_template_with_no_parent(
        self, networks_api, mock_session, caplog
    ):
        """Test set_guest_password falls back to the literal template with no parent.

        The write is live-verified, so it must not carry the
        uncharacterised-write warning.
        """
        mock_session.request.return_value = create_mock_response(
            200, {"meta": {"code": 200}, "data": {}}
        )

        with caplog.at_level(logging.WARNING):
            await networks_api.set_guest_password("network_123", "placeholder-password")

        call_args = mock_session.request.call_args
        assert call_args.args[0] == "PUT"
        assert call_args.args[1].endswith("/2.2/networks/network_123/guestnetwork/password")
        assert call_args.kwargs["data"] == {"password": "placeholder-password"}
        assert not any("not been fully characterised" in m for m in caplog.messages)

    @pytest.mark.asyncio
    async def test_set_guest_password_prefers_guest_parent_link(self, networks_api, mock_session):
        """Test set_guest_password uses the guest envelope's own password link."""
        mock_session.request.return_value = create_mock_response(
            200, {"meta": {"code": 200}, "data": {}}
        )
        guest_parent = {
            "resources": {"password": "/2.3/networks/network_123/guestnetwork/password"}
        }
        guest_parent_copy = copy.deepcopy(guest_parent)

        await networks_api.set_guest_password(
            "network_123", "placeholder-password", parent=guest_parent
        )

        call_args = mock_session.request.call_args
        assert call_args.args[1].endswith("/2.3/networks/network_123/guestnetwork/password")
        assert guest_parent == guest_parent_copy

    @pytest.mark.asyncio
    async def test_clear_guest_password_deletes_password_link(
        self, networks_api, mock_session, caplog
    ):
        """Test clear_guest_password DELETEs the guest password link without warning.

        The write is live-verified, so it must not carry the
        uncharacterised-write warning.
        """
        mock_session.request.return_value = create_mock_response(
            200, {"meta": {"code": 200}, "data": {}}
        )

        with caplog.at_level(logging.WARNING):
            await networks_api.clear_guest_password("network_123")

        call_args = mock_session.request.call_args
        assert call_args.args[0] == "DELETE"
        assert call_args.args[1].endswith("/2.2/networks/network_123/guestnetwork/password")
        assert not any("not been fully characterised" in m for m in caplog.messages)


class TestNetworksAPIPasswordValidation:
    """Tests that unintended password values never reach the form body."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", [method for method, _ in PASSWORD_WRITES])
    @pytest.mark.parametrize("password", INVALID_PASSWORDS)
    async def test_invalid_password_sends_no_request(
        self, networks_api, mock_session, method, password
    ):
        """Test a non-string or empty password is rejected before auth and transport.

        Form-encoding would otherwise turn None, False or 0 into the literal text
        "None", "False" or "0" and set that as the Wi-Fi password.
        """
        with pytest.raises(EeroValidationException, match="password.*non-empty string"):
            await getattr(networks_api, method)("network_123", password)

        networks_api._auth_api.get_auth_token.assert_not_awaited()
        mock_session.request.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method,suffix", PASSWORD_WRITES)
    @pytest.mark.parametrize("password", VALID_PASSWORDS)
    async def test_valid_password_is_forwarded_unchanged(
        self, networks_api, mock_session, method, suffix, password
    ):
        """Test a string password, even "None" or space-padded, is sent verbatim as form data."""
        mock_session.request.return_value = create_mock_response(200, api_success_response({}))

        await getattr(networks_api, method)("network_123", password)

        networks_api._auth_api.get_auth_token.assert_awaited_once()
        call_args = mock_session.request.call_args
        assert call_args.args == (
            "PUT",
            f"https://api-user.e2ro.com/2.2/networks/network_123/{suffix}",
        )
        assert call_args.kwargs["data"] == {"password": password}
        assert "json" not in call_args.kwargs


class TestNetworksAPINameValidation:
    """Tests that unintended network and guest names never reach the form body."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", INVALID_NETWORK_NAMES)
    async def test_invalid_network_name_sends_no_request(self, networks_api, mock_session, name):
        """Test a non-string or empty name is rejected before auth and transport."""
        with pytest.raises(EeroValidationException, match="name.*non-empty string"):
            await networks_api.set_network_name("network_123", name)

        networks_api._auth_api.get_auth_token.assert_not_awaited()
        mock_session.request.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("enabled", [True, False])
    @pytest.mark.parametrize("name", INVALID_GUEST_NAMES)
    async def test_invalid_guest_name_sends_no_request(
        self, networks_api, mock_session, enabled, name
    ):
        """Test a supplied non-string or empty guest name is rejected before auth and transport."""
        with pytest.raises(EeroValidationException, match="name.*non-empty string"):
            await networks_api.set_guest_network("network_123", enabled=enabled, name=name)

        networks_api._auth_api.get_auth_token.assert_not_awaited()
        mock_session.request.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda api: api.set_network_name("network_123", None), id="network-none"),
            pytest.param(
                lambda api: api.set_guest_network("network_123", enabled=True, name=False),
                id="guest-bool",
            ),
        ],
    )
    async def test_invalid_name_never_reaches_the_wire(self, networks_api, mock_session, call):
        """Test an invalid name is not form-encoded into a literal SSID.

        aiohttp encodes ``{"name": None}`` as ``name=None`` (and ``False`` as
        ``name=False``), so an unguarded call renames the network to that text.
        That the call raises is covered above; this pins what the wire carries.
        """
        with contextlib.suppress(EeroValidationException):
            await call(networks_api)

        assert _form_fields_sent(mock_session) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(
                lambda api: api.set_network_name("network_123", b"leaked-ssid"), id="network"
            ),
            pytest.param(
                lambda api: api.set_guest_network("network_123", enabled=True, name=b"leaked-ssid"),
                id="guest",
            ),
        ],
    )
    async def test_rejection_does_not_echo_the_value(self, networks_api, call):
        """Test the exception names the field but never repeats the rejected value."""
        with pytest.raises(EeroValidationException) as excinfo:
            await call(networks_api)

        assert excinfo.value.field == "name"
        assert "leaked-ssid" not in str(excinfo.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", VALID_NAMES)
    async def test_valid_network_name_is_forwarded_unchanged(
        self, networks_api, mock_session, name
    ):
        """Test a string name, even "None" or space-padded, is sent verbatim as form data."""
        mock_session.request.return_value = create_mock_response(200, api_success_response({}))

        await networks_api.set_network_name("network_123", name)

        networks_api._auth_api.get_auth_token.assert_awaited_once()
        call_args = mock_session.request.call_args
        assert call_args.args == (
            "PUT",
            "https://api-user.e2ro.com/2.2/networks/network_123/settings",
        )
        assert call_args.kwargs["data"] == {"name": name}
        assert "json" not in call_args.kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize("enabled,wire_enabled", [(True, "true"), (False, "false")])
    @pytest.mark.parametrize("name", VALID_NAMES)
    async def test_valid_guest_name_is_forwarded_unchanged(
        self, networks_api, mock_session, enabled, wire_enabled, name
    ):
        """Test a string guest name, even "None" or space-padded, is sent verbatim."""
        mock_session.request.return_value = create_mock_response(200, api_success_response({}))

        await networks_api.set_guest_network("network_123", enabled=enabled, name=name)

        networks_api._auth_api.get_auth_token.assert_awaited_once()
        call_args = mock_session.request.call_args
        assert call_args.args == (
            "PUT",
            "https://api-user.e2ro.com/2.2/networks/network_123/guestnetwork",
        )
        assert call_args.kwargs["data"] == {"enabled": wire_enabled, "name": name}
        assert "json" not in call_args.kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize("enabled,wire_enabled", [(True, "true"), (False, "false")])
    @pytest.mark.parametrize(
        "name_kwargs",
        [pytest.param({"name": None}, id="none"), pytest.param({}, id="omitted")],
    )
    async def test_guest_name_none_or_omitted_is_left_out(
        self, networks_api, mock_session, enabled, wire_enabled, name_kwargs
    ):
        """Test name=None, or no name at all, still sends only the enabled field."""
        mock_session.request.return_value = create_mock_response(200, api_success_response({}))

        await networks_api.set_guest_network("network_123", enabled=enabled, **name_kwargs)

        call_args = mock_session.request.call_args
        assert call_args.args == (
            "PUT",
            "https://api-user.e2ro.com/2.2/networks/network_123/guestnetwork",
        )
        assert call_args.kwargs["data"] == {"enabled": wire_enabled}
        assert "json" not in call_args.kwargs
