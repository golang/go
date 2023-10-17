// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !darwin || !cgo || internal

package cgotest

import "testing"

func test21897(t *testing.T) {
	t.Skip("test runs only on darwin+cgo")
}
