// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

package cgotest

import "testing"

func test8517(t *testing.T) {
	t.Skip("skipping windows only test")
}
