// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cgo

package cgotlstest

import "testing"

func testTLS(t *testing.T) {
	t.Skip("cgo not supported")
}
