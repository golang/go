// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"fmt"
	"path/filepath"
)

// AppendPWD returns the result of appending PWD=dir to the environment base.
//
// The resulting environment makes os.Getwd more efficient for a subprocess
// running in dir.
func AppendPWD(base []string, dir string) []string {
	// POSIX requires PWD to be absolute.
	// Internally we only use absolute paths, so dir should already be absolute.
	if !filepath.IsAbs(dir) {
		panic(fmt.Sprintf("AppendPWD with relative path %q", dir))
	}
	return append(base, "PWD="+dir)
}
