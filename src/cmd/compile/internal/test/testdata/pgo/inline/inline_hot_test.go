// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// WARNING: Please avoid updating this file. If this file needs to be updated,
// then a new inline_hot.pprof file should be generated:
//
//  $ cd $GOROOT/src/cmd/compile/internal/test/testdata/pgo/inline/
//  $ go test -bench=. -cpuprofile ./inline_hot.pprof
package main

import "testing"

func BenchmarkA(b *testing.B) {
	benchmarkB(b)
}
func benchmarkB(b *testing.B) {

	for i := 0; true; {
		A()
		i = i + 1
		if i >= b.N {
			break
		}
		A()
		i = i + 1
		if i >= b.N {
			break
		}
		A()
		i = i + 1
		if i >= b.N {
			break
		}
		A()
		i = i + 1
		if i >= b.N {
			break
		}
		A()
		i = i + 1
		if i >= b.N {
			break
		}
		A()
		i = i + 1
		if i >= b.N {
			break
		}
	}
}
