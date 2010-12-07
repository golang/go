// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"os"
	"strings"
)

func canExec(file string) bool {
	d, err := os.Stat(file)
	if err != nil {
		return false
	}
	return d.IsRegular() && d.Permission()&0111 != 0
}

// LookPath searches for an executable binary named file
// in the directories named by the PATH environment variable.
// If file contains a slash, it is tried directly and the PATH is not consulted.
func LookPath(file string) (string, os.Error) {
	// NOTE(rsc): I wish we could use the Plan 9 behavior here
	// (only bypass the path if file begins with / or ./ or ../)
	// but that would not match all the Unix shells.

	if strings.Contains(file, "/") {
		if canExec(file) {
			return file, nil
		}
		return "", &os.PathError{"lookpath", file, os.ENOENT}
	}
	pathenv := os.Getenv("PATH")
	for _, dir := range strings.Split(pathenv, ":", -1) {
		if dir == "" {
			// Unix shell semantics: path element "" means "."
			dir = "."
		}
		if canExec(dir + "/" + file) {
			return dir + "/" + file, nil
		}
	}
	return "", &os.PathError{"lookpath", file, os.ENOENT}
}
