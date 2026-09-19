// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits

// This package is dependency-restricted; see x/tools/go/gcexportdata.TestDeps.
import "fmt"

func assert(b bool) {
	if !b {
		panic("assertion failed")
	}
}

func panicf(format string, args ...any) {
	panic(fmt.Errorf(format, args...))
}
