// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && (unix || windows)

package cgo_test

import (
	"runtime"
	"runtime/cgo"
	"testing"
)

// myLib is declared outside package cgo, as a //cgo:dynload type would be.
type myLib struct {
	cgo.Library
	name string // user fields are allowed
}

func TestOpenEmbedded(t *testing.T) {
	var path string
	switch runtime.GOOS {
	case "linux":
		path = "libc.so.6"
	case "darwin", "ios":
		path = "/usr/lib/libSystem.B.dylib"
	case "windows":
		path = "msvcrt.dll"
	default:
		t.Skipf("no known C library path on %s", runtime.GOOS)
	}
	// Open infers PT = *myLib; the constraint is satisfied by the
	// unexported method promoted from the embedded cgo.Library.
	lib, err := cgo.Open[myLib](path)
	if err != nil {
		t.Fatal(err)
	}
	var _ *myLib = lib
}
