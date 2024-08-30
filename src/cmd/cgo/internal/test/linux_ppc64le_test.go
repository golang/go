// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64le && linux && cgo

package cgotest

import "testing"

func TestPPC64CallStubs(t *testing.T) {
	testPPC64CallStubs(t)
}
