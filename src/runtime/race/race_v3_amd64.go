//go:build none
// +build none

package race

import _ "runtime/race/internal/amd64v3"

// Note: the build line above will eventually be something
// like go:build linux && amd64.v3 || darwin && amd64.v3 || ...
// as we build v3 versions for each OS.
