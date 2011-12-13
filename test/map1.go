// errchk $G -e $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {}

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
)
