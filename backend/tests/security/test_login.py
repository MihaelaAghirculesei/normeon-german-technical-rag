"""POST /api/v1/auth/login against real users: the token carries the
user's tenant and role, and a failed login says nothing about whether the
account exists."""

from fastapi.testclient import TestClient

from app.core.security import decode_access_token
from tests.security.conftest import PASSWORD, TENANT_B, login


def test_login_issues_a_token_for_the_users_tenant_and_role(client: TestClient) -> None:
    response = client.post("/api/v1/auth/login", json={"email": "ben@b.test", "password": PASSWORD})

    assert response.status_code == 200
    body = response.json()
    auth = decode_access_token(body["access_token"])
    assert (auth.tenant_id, auth.role, auth.email) == (TENANT_B, "reader", "ben@b.test")
    assert body["user"]["tenant_name"] == "B"


def test_login_ignores_case_and_surrounding_spaces_in_the_email(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login", json={"email": "  Ben@B.Test ", "password": PASSWORD}
    )

    assert response.status_code == 200


def test_a_wrong_password_and_an_unknown_email_fail_the_same_way(client: TestClient) -> None:
    wrong_password = client.post(
        "/api/v1/auth/login", json={"email": "ben@b.test", "password": "falsch"}
    )
    unknown_email = client.post(
        "/api/v1/auth/login", json={"email": "niemand@b.test", "password": PASSWORD}
    )

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()


def test_me_describes_the_signed_in_user(client: TestClient) -> None:
    response = client.get("/api/v1/auth/me", headers=login(client, "anna@a.test"))

    assert response.status_code == 200
    assert response.json()["email"] == "anna@a.test"
    assert response.json()["role"] == "admin"
