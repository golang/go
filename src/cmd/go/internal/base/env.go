// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

// AppendPWD returns the result of appending PWD=dir to the environment base.
//
// The resulting environment makes os.Getwd more efficient for a subprocess
// running in dir.
func AppendPWD(base []string, dir string) []string {
	// Internally we only use absolute paths, so dir is absolute.
	// Even if dir is not absolute, no harm done.
	return append(base, "PWD="+dir)
}
