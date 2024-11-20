// errorcheck -0 -d=ssa/check_bce/debug=1

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x

func Found(x []string) string {
	switch len(x) {
	default:
		return x[0]
	case 0, 1:
		return ""
	}
}

func NotFound(x []string) string {
	switch len(x) {
	default:
		return x[0]
	case 0:
		return ""
	case 1:
		return ""
	}
}
