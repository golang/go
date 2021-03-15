// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !compiler_bootstrap
// +build !compiler_bootstrap

package strconv

import "internal/bytealg"

// contains reports whether the string contains the byte c.
func contains(s string, c byte) bool {
	return bytealg.IndexByteString(s, c) != -1
}
