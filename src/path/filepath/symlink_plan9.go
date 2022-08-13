// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

import (
	"os"
	"strings"
	"syscall"
)

func evalSymlinks(path string) (string, error) {
	// Plan 9 doesn't have symbolic links, so no need for substitutions.
	if len(path) > 0 {
		// Check validity of path
		_, err := os.Lstat(path)
		if err != nil {
			// Return the same error value as on other operating systems
			if strings.HasSuffix(err.Error(), "not a directory") {
				err = syscall.ENOTDIR
			}
			return "", err
		}
	}
	return Clean(path), nil
}
