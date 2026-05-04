// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hidden

import (
	"path/filepath"
	"strings"
)

// IsHidden checks if a path is hidden or not in the filesystem.
// such as: '.' files in unix filesystem
func IsHidden(path string) (flag bool, err error) {

	flag = strings.HasPrefix(filepath.Base(path), ".")

	return
}
