# ADR 0004: Tenant isolation, with row-level security as the second layer

## Status
Accepted

## Context
Several tenants share one database. A tenant must never see another
tenant's documents, chunks or answers, and must not even learn that they
exist: a question answered only by another tenant's documents has to come
back as `NICHT_GEFUNDEN`, not as a 403.

Until Giorno 25 the tenant was a request parameter (`tenant_id` in the body
or the query string) and every query filtered on it. That has two
weaknesses:
- **The client chose the tenant.** Anyone could ask on behalf of any tenant.
- **Isolation rested on every query remembering its filter.** One missing
  `WHERE tenant_id = …`, in a new endpoint or a refactored query, leaks
  data, and nothing would notice.

## Decision
1. **The tenant comes from the access token, and only from there.** Login
   issues a short-lived HS256 JWT carrying user id, tenant id and role. No
   endpoint accepts a tenant from the client; a `tenant_id` sent in a body
   is ignored. Passwords are hashed with scrypt (standard library).
2. **The application keeps filtering on the tenant** in every query: the
   first layer.
3. **Row-level security on `documents`, `chunks` and `query_logs`** is the
   second layer. Each policy matches
   `tenant_id = current_setting('app.tenant_id', true)::uuid`, for reads
   (`USING`) and for writes (`WITH CHECK`).
4. **RLS needs a role it actually applies to.** Superusers and `BYPASSRLS`
   roles ignore every policy, and the role the app connects as is one of
   them: the Postgres image's `POSTGRES_USER` is a superuser. Policies
   alone would therefore protect nothing, while every test still passed.
   Migration 0004 creates `normeon_app` (no superuser, no `BYPASSRLS`, not
   the table owner). Every transaction of a tenant-scoped session starts
   with `SET LOCAL ROLE normeon_app` and
   `set_config('app.tenant_id', …, true)`. This is a SQLAlchemy
   `after_begin` listener, so it is re-applied after each mid-request
   commit and ends with the transaction: a pooled connection never carries
   a tenant over to the next request.
5. **Missing context fails closed.** Without `app.tenant_id` the setting
   reads as NULL and the policy matches no row.

## Consequences
- **What the layers are.** RLS is the seatbelt; the application's filters
  are the braces. Either alone keeps tenants apart. `tests/security`
  checks each one separately with mutation runs:
  - with the scoping listener disabled, the RLS tests fail;
  - with an application filter removed, RLS still isolates;
  - with both off, the HTTP isolation tests catch the leak.
- **What runs without RLS.** Login, health, migrations and the maintenance
  scripts (corpus ingestion, evaluation) use an unscoped session as the
  connecting role. That is deliberate: they act across tenants or before
  a tenant is known. Request handlers that touch tenant data cannot get
  an unscoped session: `TenantSession` requires a valid token.
- **`embeddings` has no tenant column** and no policy. Every read path
  joins `chunks`, which is covered. A policy that looks up `chunks` for
  each row would sit in the vector search's hot path.
- **Production.** Production connects with a dedicated login role that is
  a member of `normeon_app`, not a superuser. The migration grants
  membership to the migrating role. Startup refuses the default JWT secret
  outside development and test.
- **Token handling.** The frontend keeps the token in `sessionStorage`,
  which is readable by script and therefore exposed to XSS. The
  mitigations are short token lifetimes and Angular's template escaping.
  The upgrade path is an httpOnly cookie set by the backend, together with
  CSRF protection.
