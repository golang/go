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

// Devirtualization of callees from transitive dependencies should work even if
// they aren't directly referenced in the package. See #61577.
//
// Dots in the last package path component are escaped in symbol names. Use one
// to ensure the escaping doesn't break lookup.
import (
	"fmt"

	"example.com/pgo/devirtualize/mult.pkg"
)

var sink int

type Adder interface {
	Add(a, b int) int
}

type Add struct{}

func (Add) Add(a, b int) int {
	for i := 0; i < 1000; i++ {
		sink++
	}
	return a + b
}

type Sub struct{}

func (Sub) Add(a, b int) int {
	for i := 0; i < 1000; i++ {
		sink++
	}
	return a - b
}

// ExerciseIface calls mostly a1 and m1.
//
//go:noinline
func ExerciseIface(iter int, a1, a2 Adder, m1, m2 mult.Multiplier) int {
	// The call below must evaluate selectA() to determine the receiver to
	// use. This should happen exactly once per iteration. Assert that is
	// the case to ensure the IR manipulation does not result in over- or
	// under-evaluation.
	selectI := 0
	selectA := func(gotI int) Adder {
		if gotI != selectI {
			panic(fmt.Sprintf("selectA not called once per iteration; got i %d want %d", gotI, selectI))
		}
		selectI++

		if gotI%10 == 0 {
			return a2
		}
		return a1
	}
	oneI := 0
	one := func(gotI int) int {
		if gotI != oneI {
			panic(fmt.Sprintf("one not called once per iteration; got i %d want %d", gotI, oneI))
		}
		oneI++

		// The function value must be evaluated before arguments, so
		// selectI must have been incremented already.
		if selectI != oneI {
			panic(fmt.Sprintf("selectA not called before not called before one; got i %d want %d", selectI, oneI))
		}

		return 1
	}

	val := 0
	for i := 0; i < iter; i++ {
		m := m1
		if i%10 == 0 {
			m = m2
		}

		// N.B. Profiles only distinguish calls on a per-line level,
		// making the two calls ambiguous. However because the
		// interfaces and implementations are mutually exclusive,
		// devirtualization can still select the correct callee for
		// each.
		//
		// If they were not mutually exclusive (for example, two Add
		// calls), then we could not definitively select the correct
		// callee.
		val += m.Multiply(42, selectA(i).Add(one(i), 2))
	}
	return val
}

type AddFunc func(int, int) int

func AddFn(a, b int) int {
	for i := 0; i < 1000; i++ {
		sink++
	}
	return a + b
}

func SubFn(a, b int) int {
	for i := 0; i < 1000; i++ {
		sink++
	}
	return a - b
}

// ExerciseFuncConcrete calls mostly a1 and m1.
//
//go:noinline
func ExerciseFuncConcrete(iter int, a1, a2 AddFunc, m1, m2 mult.MultFunc) int {
	// The call below must evaluate selectA() to determine the function to
	// call. This should happen exactly once per iteration. Assert that is
	// the case to ensure the IR manipulation does not result in over- or
	// under-evaluation.
	selectI := 0
	selectA := func(gotI int) AddFunc {
		if gotI != selectI {
			panic(fmt.Sprintf("selectA not called once per iteration; got i %d want %d", gotI, selectI))
		}
		selectI++

		if gotI%10 == 0 {
			return a2
		}
		return a1
	}
	oneI := 0
	one := func(gotI int) int {
		if gotI != oneI {
			panic(fmt.Sprintf("one not called once per iteration; got i %d want %d", gotI, oneI))
		}
		oneI++

		// The function value must be evaluated before arguments, so
		// selectI must have been incremented already.
		if selectI != oneI {
			panic(fmt.Sprintf("selectA not called before not called before one; got i %d want %d", selectI, oneI))
		}

		return 1
	}

	val := 0
	for i := 0; i < iter; i++ {
		m := m1
		if i%10 == 0 {
			m = m2
		}

		// N.B. Profiles only distinguish calls on a per-line level,
		// making the two calls ambiguous. However because the
		// function types are mutually exclusive, devirtualization can
		// still select the correct callee for each.
		//
		// If they were not mutually exclusive (for example, two
		// AddFunc calls), then we could not definitively select the
		// correct callee.
		val += int(m(42, int64(selectA(i)(one(i), 2))))
	}
	return val
}

// ExerciseFuncField calls mostly a1 and m1.
//
// This is a simplified version of ExerciseFuncConcrete, but accessing the
// function values via a struct field.
//
//go:noinline
func ExerciseFuncField(iter int, a1, a2 AddFunc, m1, m2 mult.MultFunc) int {
	ops := struct {
		a AddFunc
		m mult.MultFunc
	}{}

	val := 0
	for i := 0; i < iter; i++ {
		ops.a = a1
		ops.m = m1
		if i%10 == 0 {
			ops.a = a2
			ops.m = m2
		}

		// N.B. Profiles only distinguish calls on a per-line level,
		// making the two calls ambiguous. However because the
		// function types are mutually exclusive, devirtualization can
		// still select the correct callee for each.
		//
		// If they were not mutually exclusive (for example, two
		// AddFunc calls), then we could not definitively select the
		// correct callee.
		val += int(ops.m(42, int64(ops.a(1, 2))))
	}
	return val
}

//go:noinline
func AddClosure() AddFunc {
	// Implicit closure by capturing the receiver.
	var a Add
	return a.Add
}

//go:noinline
func SubClosure() AddFunc {
	var s Sub
	return s.Add
}

// ExerciseFuncClosure calls mostly a1 and m1.
//
// This is a simplified version of ExerciseFuncConcrete, but we need two
// distinct call sites to test two different types of function values.
//
//go:noinline
func ExerciseFuncClosure(iter int, a1, a2 AddFunc, m1, m2 mult.MultFunc) int {
	val := 0
	for i := 0; i < iter; i++ {
		a := a1
		m := m1
		if i%10 == 0 {
			a = a2
			m = m2
		}

		// N.B. Profiles only distinguish calls on a per-line level,
		// making the two calls ambiguous. However because the
		// function types are mutually exclusive, devirtualization can
		// still select the correct callee for each.
		//
		// If they were not mutually exclusive (for example, two
		// AddFunc calls), then we could not definitively select the
		// correct callee.
		val += int(m(42, int64(a(1, 2))))
	}
	return val
}

//go:noinline
func IfaceZeroWeight(a *Add, b Adder) bool {
	return a.Add(1, 2) == b.Add(3, 4) // unwanted devirtualization
}

// ExerciseIfaceZeroWeight never calls IfaceZeroWeight, so the callee
// is not expected to appear in the profile.
//
//go:noinline
func ExerciseIfaceZeroWeight() {
	if false {
		a := &Add{}
		b := &Sub{}
		// Unreachable call
		IfaceZeroWeight(a, b)
	}
}

func DirectCall() bool {
	return true
}

func IndirectCall() bool {
	return false
}

//go:noinline
func IndirCallZeroWeight(indirectCall func() bool) bool {
	return DirectCall() && indirectCall() // unwanted devirtualization
}

// ExerciseIndirCallZeroWeight never calls IndirCallZeroWeight, so the
// callee is not expected to appear in the profile.
//
//go:noinline
func ExerciseIndirCallZeroWeight() {
	if false {
		// Unreachable call
		IndirCallZeroWeight(IndirectCall)
	}
}
