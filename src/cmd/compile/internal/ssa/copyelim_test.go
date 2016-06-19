// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"testing"
)

func BenchmarkCopyElim1(b *testing.B)      { benchmarkCopyElim(b, 1) }
func BenchmarkCopyElim10(b *testing.B)     { benchmarkCopyElim(b, 10) }
func BenchmarkCopyElim100(b *testing.B)    { benchmarkCopyElim(b, 100) }
func BenchmarkCopyElim1000(b *testing.B)   { benchmarkCopyElim(b, 1000) }
func BenchmarkCopyElim10000(b *testing.B)  { benchmarkCopyElim(b, 10000) }
func BenchmarkCopyElim100000(b *testing.B) { benchmarkCopyElim(b, 100000) }

func benchmarkCopyElim(b *testing.B, n int) {
	c := testConfig(b)

	values := make([]interface{}, 0, n+2)
	values = append(values, Valu("mem", OpInitMem, TypeMem, 0, nil))
	last := "mem"
	for i := 0; i < n; i++ {
		name := fmt.Sprintf("copy%d", i)
		values = append(values, Valu(name, OpCopy, TypeMem, 0, nil, last))
		last = name
	}
	values = append(values, Exit(last))
	// Reverse values array to make it hard
	for i := 0; i < len(values)/2; i++ {
		values[i], values[len(values)-1-i] = values[len(values)-1-i], values[i]
	}

	for i := 0; i < b.N; i++ {
		fun := Fun(c, "entry", Bloc("entry", values...))
		Copyelim(fun.f)
		fun.f.Free()
	}
}
