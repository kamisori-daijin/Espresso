#!/bin/bash

set -euo pipefail

APP_NAME="EspressoBench"
APP_PRODUCT="espresso-bench-app"
CLI_PRODUCT="espresso-bench"
APPS_DIR=".build/apps"
OUTPUT_DIR="${APPS_DIR}/${APP_NAME}.app"
STAGING_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/espresso-bench-app.XXXXXX")"
STAGING_APP="${STAGING_ROOT}/${APP_NAME}.app"
CONTENTS_DIR="${STAGING_APP}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
trap 'rm -rf "${STAGING_ROOT}"' EXIT

swift build -c release --product "${APP_PRODUCT}"
swift build -c release --product "${CLI_PRODUCT}"
BUILD_DIR="$(swift build -c release --show-bin-path)"

mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

cp "${BUILD_DIR}/${APP_PRODUCT}" "${MACOS_DIR}/EspressoBenchApp"
cp "${BUILD_DIR}/${CLI_PRODUCT}" "${MACOS_DIR}/espresso-bench"

cat > "${CONTENTS_DIR}/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>EspressoBenchApp</string>
    <key>CFBundleIdentifier</key>
    <string>org.espresso.bench.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>EspressoBench</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

mkdir -p "${APPS_DIR}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  BACKUP_DIR="${OUTPUT_DIR}.previous.$(date +%Y%m%d-%H%M%S)"
  mv "${OUTPUT_DIR}" "${BACKUP_DIR}"
  echo "Moved existing bundle to ${BACKUP_DIR}"
fi

mv "${STAGING_APP}" "${OUTPUT_DIR}"
echo "Packaged app at ${OUTPUT_DIR}"
