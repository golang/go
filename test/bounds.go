// errorcheck -0 -m -l

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test, using compiler diagnostic flags, that bounds check elimination
// is eliminating the correct checks.

package foo

var (
	s []int

	a1 [1]int
	a1k [1000]int
	a100k [100000]int

	p1 *[1]int
	p1k *[1000]int
	p100k *[100000]int

	i int
	ui uint
	i8 int8
	ui8 uint8
	i16 int16
	ui16 uint16
	i32 int32
	ui32 uint32
	i64 int64
	ui64 uint64
)

func main() {
	// Most things need checks.
	use(s[i])
	use(a1[i])
	use(a1k[i])
	use(a100k[i])
	use(p1[i])
	use(p1k[i])
	use(p100k[i])

	use(s[ui])
	use(a1[ui])
	use(a1k[ui])
	use(a100k[ui])
	use(p1[ui])
	use(p1k[ui])
	use(p100k[ui])

	use(s[i8])
	use(a1[i8])
	use(a1k[i8])
	use(a100k[i8])
	use(p1[i8])
	use(p1k[i8])
	use(p100k[i8])

	// Unsigned 8-bit numbers don't need checks for len >= 2⁸.
	use(s[ui8])
	use(a1[ui8])
	use(a1k[ui8])  // ERROR "index bounds check elided"
	use(a100k[ui8])  // ERROR "index bounds check elided"
	use(p1[ui8])
	use(p1k[ui8])  // ERROR "index bounds check elided"
	use(p100k[ui8])  // ERROR "index bounds check elided"

	use(s[i16])
	use(a1[i16])
	use(a1k[i16])
	use(a100k[i16])
	use(p1[i16])
	use(p1k[i16])
	use(p100k[i16])

	// Unsigned 16-bit numbers don't need checks for len >= 2¹⁶.
	use(s[ui16])
	use(a1[ui16])
	use(a1k[ui16])
	use(a100k[ui16])  // ERROR "index bounds check elided"
	use(p1[ui16])
	use(p1k[ui16])
	use(p100k[ui16])  // ERROR "index bounds check elided"

	use(s[i32])
	use(a1[i32])
	use(a1k[i32])
	use(a100k[i32])
	use(p1[i32])
	use(p1k[i32])
	use(p100k[i32])

	use(s[ui32])
	use(a1[ui32])
	use(a1k[ui32])
	use(a100k[ui32])
	use(p1[ui32])
	use(p1k[ui32])
	use(p100k[ui32])

	use(s[i64])
	use(a1[i64])
	use(a1k[i64])
	use(a100k[i64])
	use(p1[i64])
	use(p1k[i64])
	use(p100k[i64])

	use(s[ui64])
	use(a1[ui64])
	use(a1k[ui64])
	use(a100k[ui64])
	use(p1[ui64])
	use(p1k[ui64])
	use(p100k[ui64])

	// Mod truncates the maximum value to one less than the argument,
	// but signed mod can be negative, so only unsigned mod counts.
	use(s[i%999])
	use(a1[i%999])
	use(a1k[i%999])
	use(a100k[i%999])
	use(p1[i%999])
	use(p1k[i%999])
	use(p100k[i%999])

	use(s[ui%999])
	use(a1[ui%999])
	use(a1k[ui%999])  // ERROR "index bounds check elided"
	use(a100k[ui%999])  // ERROR "index bounds check elided"
	use(p1[ui%999])
	use(p1k[ui%999])  // ERROR "index bounds check elided"
	use(p100k[ui%999])  // ERROR "index bounds check elided"

	use(s[i%1000])
	use(a1[i%1000])
	use(a1k[i%1000])
	use(a100k[i%1000])
	use(p1[i%1000])
	use(p1k[i%1000])
	use(p100k[i%1000])

	use(s[ui%1000])
	use(a1[ui%1000])
	use(a1k[ui%1000])  // ERROR "index bounds check elided"
	use(a100k[ui%1000])  // ERROR "index bounds check elided"
	use(p1[ui%1000])
	use(p1k[ui%1000])  // ERROR "index bounds check elided"
	use(p100k[ui%1000])  // ERROR "index bounds check elided"

	use(s[i%1001])
	use(a1[i%1001])
	use(a1k[i%1001])
	use(a100k[i%1001])
	use(p1[i%1001])
	use(p1k[i%1001])
	use(p100k[i%1001])

	use(s[ui%1001])
	use(a1[ui%1001])
	use(a1k[ui%1001])
	use(a100k[ui%1001])  // ERROR "index bounds check elided"
	use(p1[ui%1001])
	use(p1k[ui%1001])
	use(p100k[ui%1001])  // ERROR "index bounds check elided"

	// Bitwise and truncates the maximum value to the mask value.
	// The result (for a positive mask) cannot be negative, so elision
	// applies to both signed and unsigned indexes.
	use(s[i&999])
	use(a1[i&999])
	use(a1k[i&999])  // ERROR "index bounds check elided"
	use(a100k[i&999])  // ERROR "index bounds check elided"
	use(p1[i&999])
	use(p1k[i&999])  // ERROR "index bounds check elided"
	use(p100k[i&999])  // ERROR "index bounds check elided"

	use(s[ui&999])
	use(a1[ui&999])
	use(a1k[ui&999])  // ERROR "index bounds check elided"
	use(a100k[ui&999])  // ERROR "index bounds check elided"
	use(p1[ui&999])
	use(p1k[ui&999])  // ERROR "index bounds check elided"
	use(p100k[ui&999])  // ERROR "index bounds check elided"

	use(s[i&1000])
	use(a1[i&1000])
	use(a1k[i&1000])
	use(a100k[i&1000])  // ERROR "index bounds check elided"
	use(p1[i&1000])
	use(p1k[i&1000])
	use(p100k[i&1000])  // ERROR "index bounds check elided"

	use(s[ui&1000])
	use(a1[ui&1000])
	use(a1k[ui&1000])
	use(a100k[ui&1000])  // ERROR "index bounds check elided"
	use(p1[ui&1000])
	use(p1k[ui&1000])
	use(p100k[ui&1000])  // ERROR "index bounds check elided"

	// Right shift cuts the effective number of bits in the index,
	// but only for unsigned (signed stays negative).
	use(s[i32>>22])
	use(a1[i32>>22])
	use(a1k[i32>>22])
	use(a100k[i32>>22])
	use(p1[i32>>22])
	use(p1k[i32>>22])
	use(p100k[i32>>22])

	use(s[ui32>>22])
	use(a1[ui32>>22])
	use(a1k[ui32>>22])
	use(a100k[ui32>>22])  // ERROR "index bounds check elided"
	use(p1[ui32>>22])
	use(p1k[ui32>>22])
	use(p100k[ui32>>22])  // ERROR "index bounds check elided"

	use(s[i32>>23])
	use(a1[i32>>23])
	use(a1k[i32>>23])
	use(a100k[i32>>23])
	use(p1[i32>>23])
	use(p1k[i32>>23])
	use(p100k[i32>>23])

	use(s[ui32>>23])
	use(a1[ui32>>23])
	use(a1k[ui32>>23])  // ERROR "index bounds check elided"
	use(a100k[ui32>>23])  // ERROR "index bounds check elided"
	use(p1[ui32>>23])
	use(p1k[ui32>>23])  // ERROR "index bounds check elided"
	use(p100k[ui32>>23])  // ERROR "index bounds check elided"

	// Division cuts the range like right shift does.
	use(s[i/1e6])
	use(a1[i/1e6])
	use(a1k[i/1e6])
	use(a100k[i/1e6])
	use(p1[i/1e6])
	use(p1k[i/1e6])
	use(p100k[i/1e6])

	use(s[ui/1e6])
	use(a1[ui/1e6])
	use(a1k[ui/1e6])
	use(p1[ui/1e6])
	use(p1k[ui/1e6])

	use(s[i/1e7])
	use(a1[i/1e7])
	use(a1k[i/1e7])
	use(a100k[i/1e7])
	use(p1[i/1e7])
	use(p1k[i/1e7])
	use(p100k[i/1e7])

	use(s[ui/1e7])
	use(a1[ui/1e7])
	use(p1[ui/1e7])
}

var sum int 

func use(x int) {
	sum += x
}
