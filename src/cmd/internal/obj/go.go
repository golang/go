// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"math"
	"strings"
)

// go-specific code shared across loaders (5l, 6l, 8l).

// replace all "". with pkg.
func expandpkg(t0 string, pkg string) string {
	return strings.Replace(t0, `"".`, pkg+".", -1)
}

func double2ieee(ieee *uint64, f float64) {
	*ieee = math.Float64bits(f)
}
