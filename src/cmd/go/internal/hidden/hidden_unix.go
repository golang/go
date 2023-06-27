// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hidden

import (
	"path/filepath"
	"strings"
)

// IsHidden reports whether path is hidden by default in user interfaces
// on the current platform.
func IsHidden(path string) (flag bool) {

	flag = strings.HasPrefix(path, ".") || strings.HasPrefix(filepath.Base(path), ".")

	return
}
