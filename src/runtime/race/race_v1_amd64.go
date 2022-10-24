//go:build (linux && !amd64.v3) || darwin || freebsd || netbsd || openbsd || windows
// +build linux,!amd64.v3 darwin freebsd netbsd openbsd windows

package race

import _ "runtime/race/internal/amd64v1"
