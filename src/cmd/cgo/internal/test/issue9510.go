// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && !((ppc64 || ppc64le) && internal)

// Test that we can link together two different cgo packages that both
// use the same libgcc function.

package cgotest

import (
	"runtime"
	"testing"

	"cmd/cgo/internal/test/issue9510a"
	"cmd/cgo/internal/test/issue9510b"
)

func test9510(t *testing.T) {
	if runtime.GOARCH == "arm" {
		t.Skip("skipping because libgcc may be a Thumb library")
	}
	issue9510a.F(1, 1)
	issue9510b.F(1, 1)
}
