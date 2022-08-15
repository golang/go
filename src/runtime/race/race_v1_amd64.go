//go:build linux || darwin || freebsd || netbsd || openbsd || windows
// +build linux darwin freebsd netbsd openbsd windows

package race

import _ "runtime/race/internal/amd64v1"

// Note: the build line above will eventually be something
// like go:build linux && !amd64.v3 || darwin && !amd64.v3 || ...
// as we build v3 versions for each OS.
