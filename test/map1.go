// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test map declarations of many types, including erroneous ones.
// Does not compile.

package main

type v bool

var (
	// valid
	_ map[int8]v
	_ map[uint8]v
	_ map[int16]v
	_ map[uint16]v
	_ map[int32]v
	_ map[uint32]v
	_ map[int64]v
	_ map[uint64]v
	_ map[int]v
	_ map[uint]v
	_ map[uintptr]v
	_ map[float32]v
	_ map[float64]v
	_ map[complex64]v
	_ map[complex128]v
	_ map[bool]v
	_ map[string]v
	_ map[chan int]v
	_ map[*int]v
	_ map[struct{}]v
	_ map[[10]int]v

	// invalid
	_ map[[]int]v       // ERROR "invalid map key"
	_ map[func()]v      // ERROR "invalid map key"
	_ map[map[int]int]v // ERROR "invalid map key"
	_ map[T1]v    // ERROR "invalid map key"
	_ map[T2]v    // ERROR "invalid map key"
	_ map[T3]v    // ERROR "invalid map key"
	_ map[T4]v    // ERROR "invalid map key"
	_ map[T5]v
	_ map[T6]v
	_ map[T7]v
	_ map[T8]v
)

type T1 []int
type T2 struct { F T1 }
type T3 []T4
type T4 struct { F T3 }

type T5 *int
type T6 struct { F T5 }
type T7 *T4
type T8 struct { F *T7 }

func main() {
	m := make(map[int]int)
	delete()        // ERROR "missing arguments"
	delete(m)       // ERROR "missing second \(key\) argument"
	delete(m, 2, 3) // ERROR "too many arguments"
	delete(1, m)    // ERROR "first argument to delete must be map"
}