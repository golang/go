// +build !windows

package filepath

import (
	"syscall"
)

// walkSymlinks returns slashAfterFilePathError error for paths like
// //path/to/existing_file/ and /path/to/existing_file/. and /path/to/existing_file/..

var slashAfterFilePathError = syscall.ENOTDIR

func evalSymlinks(path string) (string, error) {
	return walkSymlinks(path)
}
