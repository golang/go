// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hidden

import (
	"path/filepath"
	"strings"
)

// IsHidden checks if a path is hidden or not
// Eg: .DS_Store in mac
// Eg: . in linux
func IsHidden(path string) (flag bool) {

	flag = strings.HasPrefix(path, ".") || strings.HasPrefix(filepath.Base(path), ".")

	return
}
