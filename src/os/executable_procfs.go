// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || netbsd || (js && wasm)

package os

import (
	"errors"
	"runtime"
)

func executable() (string, error) {
	var procfn string
	switch runtime.GOOS {
	default:
		return "", errors.New("Executable not implemented for " + runtime.GOOS)
	case "linux", "android":
		procfn = "/proc/self/exe"
	case "netbsd":
		procfn = "/proc/curproc/exe"
	}
	path, err := Readlink(procfn)

	// When the executable has been deleted then Readlink returns a
	// path appended with " (deleted)".
	return stringsTrimSuffix(path, " (deleted)"), err
}

// stringsTrimSuffix is the same as strings.TrimSuffix.
func stringsTrimSuffix(s, suffix string) string {
	if len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix {
		return s[:len(s)-len(suffix)]
	}
	return s
}
