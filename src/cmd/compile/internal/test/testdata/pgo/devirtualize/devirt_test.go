// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// WARNING: Please avoid updating this file. If this file needs to be updated,
// then a new devirt.pprof file should be generated:
//
//	$ cd $GOROOT/src/cmd/compile/internal/test/testdata/pgo/devirtualize/
//	$ go mod init example.com/pgo/devirtualize
//	$ go test -bench=. -cpuprofile ./devirt.pprof

package devirt

import (
	"testing"

	"example.com/pgo/devirtualize/mult.pkg"
)

func BenchmarkDevirt(b *testing.B) {
	var (
		a1 Add
		a2 Sub
		m1 mult.Mult
		m2 mult.NegMult
	)

	Exercise(b.N, a1, a2, m1, m2)
}
