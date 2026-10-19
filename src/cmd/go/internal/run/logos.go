// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package run

import (
	"strings"
)

// trimRunExeExt trims the source extension of an explicit command-line file
// to form the executable name. Upstream trims ".go"; logos files (including
// aliases) trim their actual extension, which starts at the last dot.
func trimRunExeExt(src string) string {
	if rest, found := strings.CutSuffix(src, ".go"); found {
		return rest
	}
	if stem, _, found := strings.CutLast(src, "."); found {
		return stem
	}
	return src
}
