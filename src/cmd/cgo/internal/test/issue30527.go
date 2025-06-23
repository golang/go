// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

// Issue 30527: function call rewriting casts untyped
// constants to int because of ":=" usage.

package cgotest

import "cmd/cgo/internal/test/issue30527"

func issue30527G() {
	issue30527.G(nil)
}
