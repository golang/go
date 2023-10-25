// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !compiler_bootstrap

package strconv

import "internal/bytealg"

// index returns the index of the first instance of c in s, or -1 if missing.
func index(s string, c byte) int {
	return bytealg.IndexByteString(s, c)
}
