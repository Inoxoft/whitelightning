#!/bin/bash

APP_USER="appuser"

CURRENT_CONTAINER_UID=$(id -u)
CURRENT_CONTAINER_GID=$(id -g)

APP_USER_BUILT_UID=$(id -u "${APP_USER}")
APP_USER_BUILT_GID=$(id -g "${APP_USER}")

echo "Container running with UID: ${CURRENT_CONTAINER_UID}, GID: ${CURRENT_CONTAINER_GID}"
echo "App user (${APP_USER}) built with UID: ${APP_USER_BUILT_UID}, GID: ${APP_USER_BUILT_GID}"

if [ "${CURRENT_CONTAINER_UID}" -ne "${APP_USER_BUILT_UID}" ]; then
    echo "Host user UID (${CURRENT_CONTAINER_UID}) differs from app user UID (${APP_USER_BUILT_UID})."
    echo "Adjusting ownership of mounted volumes to ${APP_USER}."
    chown -R "${APP_USER}":"${APP_USER}" /app/models /app/own_data
fi

chmod -R ug+rwx /app/models /app/own_data

echo "Executing command as user ${APP_USER}: $@"
exec gosu "${APP_USER}" "$@" 