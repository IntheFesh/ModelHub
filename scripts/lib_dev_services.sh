#!/usr/bin/env bash
# Shared helpers for scripts/quickstart.sh and scripts/showcase.sh.
#
# Starts the LOCAL, non-networked dev instances of Redis/PostgreSQL this
# repo's tests and `make demo` optionally use (tests/conftest.py's
# `requires_redis`/`requires_postgres` markers skip honestly — not "fail" —
# when these aren't reachable; CLAUDE.md's real infra, no mocks). Every
# function here is best-effort and idempotent: safe to source and call
# repeatedly, never touches anything outside this machine's local Redis/
# Postgres, and never fails the calling script — it only reports status.
#
# Not meant to be run directly; `source` it from another script.

_MODELHUB_REDIS_PORT=6379
_MODELHUB_PG_TEST_DB="modelhub_test"
_MODELHUB_PG_TEST_USER="modelhub_test"
_MODELHUB_PG_TEST_PASSWORD="modelhub_test"

_have() { command -v "$1" >/dev/null 2>&1; }

# Starts a local Redis on the default port if one isn't already answering.
# `make demo` and the gateway (A6) tests need this; nothing here touches a
# remote Redis or changes any config beyond starting the local daemon.
ensure_redis() {
    if _have redis-cli && redis-cli -p "$_MODELHUB_REDIS_PORT" ping >/dev/null 2>&1; then
        echo "  [redis]      already running on :${_MODELHUB_REDIS_PORT}"
        return 0
    fi
    if ! _have redis-server; then
        echo "  [redis]      not installed — skip (gateway/demo tests needing it will skip honestly, not fail)"
        return 1
    fi
    echo "  [redis]      starting local redis-server..."
    if _have service && service redis-server start >/tmp/modelhub-redis-start.log 2>&1; then
        :
    else
        redis-server --daemonize yes --port "$_MODELHUB_REDIS_PORT" >/tmp/modelhub-redis-start.log 2>&1 || true
    fi
    sleep 1
    if _have redis-cli && redis-cli -p "$_MODELHUB_REDIS_PORT" ping >/dev/null 2>&1; then
        echo "  [redis]      up on :${_MODELHUB_REDIS_PORT}"
        return 0
    fi
    echo "  [redis]      could not start — see /tmp/modelhub-redis-start.log (gateway/demo tests needing it will skip honestly, not fail)"
    return 1
}

# Starts a local PostgreSQL and, if needed, creates the exact
# role/password/database tests/conftest.py::TEST_PG_DSN expects
# (modelhub_test/modelhub_test/modelhub_test) — never touches any other
# role or database. Optional: only a handful of A2/A3 tests use Postgres,
# everything else in this repo runs fine without it.
ensure_postgres() {
    if ! _have pg_isready && ! _have psql; then
        echo "  [postgres]   not installed — skip (the handful of tests needing it will skip honestly, not fail)"
        return 1
    fi
    if ! pg_isready >/dev/null 2>&1; then
        echo "  [postgres]   starting local postgresql..."
        if _have service; then
            service postgresql start >/tmp/modelhub-postgres-start.log 2>&1 || true
        fi
        sleep 2
    fi
    if ! pg_isready >/dev/null 2>&1; then
        echo "  [postgres]   could not start — see /tmp/modelhub-postgres-start.log (skip, not fatal)"
        return 1
    fi
    echo "  [postgres]   up"

    local role_exists
    role_exists=$(sudo -u postgres psql -tAc \
        "SELECT 1 FROM pg_roles WHERE rolname='${_MODELHUB_PG_TEST_USER}'" 2>/dev/null || echo "")
    if [ "$role_exists" != "1" ]; then
        echo "  [postgres]   creating role/database '${_MODELHUB_PG_TEST_DB}' for tests..."
        sudo -u postgres psql -c \
            "CREATE ROLE ${_MODELHUB_PG_TEST_USER} LOGIN CREATEDB PASSWORD '${_MODELHUB_PG_TEST_PASSWORD}';" \
            >/dev/null 2>&1 || true
        sudo -u postgres psql -c \
            "CREATE DATABASE ${_MODELHUB_PG_TEST_DB} OWNER ${_MODELHUB_PG_TEST_USER};" \
            >/dev/null 2>&1 || true
    fi
    if PGPASSWORD="$_MODELHUB_PG_TEST_PASSWORD" psql -h 127.0.0.1 -U "$_MODELHUB_PG_TEST_USER" \
        -d "$_MODELHUB_PG_TEST_DB" -c "SELECT 1" >/dev/null 2>&1; then
        echo "  [postgres]   test role/database ready (${_MODELHUB_PG_TEST_DB})"
        return 0
    fi
    echo "  [postgres]   role/database setup could not be verified — see CLAUDE.md/README for manual steps (skip, not fatal)"
    return 1
}
