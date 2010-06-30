// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Unix environment variables.

package os

// TempDir returns the default directory to use for temporary files.
// On Unix-like systems, it uses the environment variable $TMPDIR
// or, if that is empty, /tmp.
// On Windows systems, it uses the Windows GetTempPath API.
func TempDir() string {
	dir := Getenv("TMPDIR")
	if dir == "" {
		dir = "/tmp"
	}
	return dir
}
