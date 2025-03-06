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

func BenchmarkDevirtIface(b *testing.B) {
	var (
		a1 Add
		a2 Sub
		m1 mult.Mult
		m2 mult.NegMult
	)

	ExerciseIface(b.N, a1, a2, m1, m2)
}

// Verify that devirtualization doesn't result in calls or side effects applying more than once.
func TestDevirtIface(t *testing.T) {
	var (
		a1 Add
		a2 Sub
		m1 mult.Mult
		m2 mult.NegMult
	)

	if v := ExerciseIface(10, a1, a2, m1, m2); v != 1176 {
		t.Errorf("ExerciseIface(10) got %d want 1176", v)
	}
}

func BenchmarkDevirtFuncConcrete(b *testing.B) {
	ExerciseFuncConcrete(b.N, AddFn, SubFn, mult.MultFn, mult.NegMultFn)
}

func TestDevirtFuncConcrete(t *testing.T) {
	if v := ExerciseFuncConcrete(10, AddFn, SubFn, mult.MultFn, mult.NegMultFn); v != 1176 {
		t.Errorf("ExerciseFuncConcrete(10) got %d want 1176", v)
	}
}

func BenchmarkDevirtFuncField(b *testing.B) {
	ExerciseFuncField(b.N, AddFn, SubFn, mult.MultFn, mult.NegMultFn)
}

func TestDevirtFuncField(t *testing.T) {
	if v := ExerciseFuncField(10, AddFn, SubFn, mult.MultFn, mult.NegMultFn); v != 1176 {
		t.Errorf("ExerciseFuncField(10) got %d want 1176", v)
	}
}

func BenchmarkDevirtFuncClosure(b *testing.B) {
	ExerciseFuncClosure(b.N, AddClosure(), SubClosure(), mult.MultClosure(), mult.NegMultClosure())
}

func TestDevirtFuncClosure(t *testing.T) {
	if v := ExerciseFuncClosure(10, AddClosure(), SubClosure(), mult.MultClosure(), mult.NegMultClosure()); v != 1176 {
		t.Errorf("ExerciseFuncClosure(10) got %d want 1176", v)
	}
}

func BenchmarkDevirtIfaceZeroWeight(t *testing.B) {
	ExerciseIfaceZeroWeight()
}

func TestDevirtIfaceZeroWeight(t *testing.T) {
	ExerciseIfaceZeroWeight()
}

func BenchmarkDevirtIndirCallZeroWeight(t *testing.B) {
	ExerciseIndirCallZeroWeight()
}

func TestDevirtIndirCallZeroWeight(t *testing.T) {
	ExerciseIndirCallZeroWeight()
}
