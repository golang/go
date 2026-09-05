// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package structs_test

import (
	"fmt"
	"structs"
	"unsafe"
)

// ExampleHostLayout demonstrates using HostLayout to match C struct layout.
func ExampleHostLayout() {
	// A struct that should match C memory layout
	type CCompatible struct {
		_ structs.HostLayout
		a int32
		b int64
		c int32
	}

	var s CCompatible
	s.a = 1
	s.b = 2
	s.c = 3

	fmt.Printf("Size: %d bytes\n", unsafe.Sizeof(s))
	fmt.Printf("Values: a=%d, b=%d, c=%d\n", s.a, s.b, s.c)

	// Output:
	// Size: 24 bytes
	// Values: a=1, b=2, c=3
}

// Example_cInterop demonstrates using HostLayout for C interoperability.
func Example_cInterop() {
	// Struct designed to match a C struct definition
	type Point struct {
		_ structs.HostLayout
		x float64
		y float64
	}

	p := Point{x: 3.14, y: 2.71}
	fmt.Printf("Point: (%.2f, %.2f)\n", p.x, p.y)

	// Output:
	// Point: (3.14, 2.71)
}

// Example_nestedStructs demonstrates that HostLayout only affects
// the immediate struct, not nested structs.
func Example_nestedStructs() {
	type Inner struct {
		value int32
	}

	type Outer struct {
		_ structs.HostLayout
		a int32
		b Inner // Inner is NOT affected by HostLayout
		c int32
	}

	var s Outer
	s.a = 1
	s.b.value = 2
	s.c = 3

	fmt.Printf("Outer size: %d bytes\n", unsafe.Sizeof(s))
	fmt.Printf("Values: a=%d, b.value=%d, c=%d\n", s.a, s.b.value, s.c)

	// Output:
	// Outer size: 12 bytes
	// Values: a=1, b.value=2, c=3
}

// Example_aliasing demonstrates that HostLayout can be aliased
// and still maintains its properties.
func Example_aliasing() {
	// Create an alias for HostLayout
	type HL structs.HostLayout

	type Data struct {
		_ HL // The alias works the same way
		x int32
		y int64
	}

	var d Data
	d.x = 100
	d.y = 200

	fmt.Printf("Size: %d bytes\n", unsafe.Sizeof(d))
	fmt.Printf("Values: x=%d, y=%d\n", d.x, d.y)

	// Output:
	// Size: 16 bytes
	// Values: x=100, y=200
}
