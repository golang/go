// run

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test division of variables. Generate many test cases,
// compute correct answer using shift and subtract,
// and then compare against results from divison and
// modulus operators.
//
// Primarily useful for testing software div/mod.

package main

const long = false

func main() {
	if long {
		// About 3e9 test cases (calls to checkdiv3).
		// Too long for everyday testing.
		gen2(3, 64, 2, 64, checkdiv1)
		println(ntest)
	} else {
		// About 4e6 test cases (calls to checkdiv3).
		// Runs for 8 seconds on ARM chromebook, much faster elsewhere.
		gen2(2, 64, 1, 64, checkdiv1)
	}
}

// generate all uint64 values x where x has at most n bits set in the low w
// and call f(x) for each.
func gen1(n, w int, f func(uint64)) {
	gen(0, 0, n, w-1, f)
}

func gen(val uint64, nbits, maxbits, pos int, f func(uint64)) {
	if pos < 0 {
		f(val)
		return
	}
	gen(val, nbits, maxbits, pos-1, f)
	if nbits < maxbits {
		gen(val|1<<uint(pos), nbits+1, maxbits, pos-1, f)
	}
}

// generate all uint64 values x, y where x has at most n1 bits set in the low w1
// and y has at most n2 bits set in the low w2 and call f(x, y) for each.
func gen2(n1, w1, n2, w2 int, f func(uint64, uint64)) {
	gen1(n1, w1, func(x uint64) {
		gen1(n2, w2, func(y uint64) {
			f(x, y)
		})
	})
}

// x and y are uint64s with at most 2 bits set.
// Check those values and values above and below,
// along with bitwise inversions of the same (done in checkdiv2).
func checkdiv1(x, y uint64) {
	checkdiv2(x, y)
	// If the low bit is set in x or y, adding or subtracting 1
	// produces a number that checkdiv1 is going to be called
	// with anyway, so don't duplicate effort.
	if x&1 == 0 {
		checkdiv2(x+1, y)
		checkdiv2(x-1, y)
	}
	if y&1 == 0 {
		checkdiv2(x, y-1)
		checkdiv2(x, y+1)
		if x&1 == 0 {
			checkdiv2(x+1, y-1)
			checkdiv2(x-1, y-1)
			checkdiv2(x-1, y+1)
			checkdiv2(x+1, y+1)
		}
	}
}

func checkdiv2(x, y uint64) {
	checkdiv3(x, y)
	checkdiv3(^x, y)
	checkdiv3(x, ^y)
	checkdiv3(^x, ^y)
}

var ntest int64 = 0

func checkdiv3(x, y uint64) {
	ntest++
	if ntest&(ntest-1) == 0 && long {
		println(ntest, "...")
	}
	checkuint64(x, y)
	if (uint64(uint32(x)) == x || uint64(uint32(^x)) == ^x) && (uint64(uint32(y)) == y || uint64(uint32(^y)) == ^y) {
		checkuint32(uint32(x), uint32(y))
	}
	if (uint64(uint16(x)) == x || uint64(uint16(^x)) == ^x) && (uint64(uint16(y)) == y || uint64(uint16(^y)) == ^y) {
		checkuint16(uint16(x), uint16(y))
	}
	if (uint64(uint8(x)) == x || uint64(uint8(^x)) == ^x) && (uint64(uint8(y)) == y || uint64(uint8(^y)) == ^y) {
		checkuint8(uint8(x), uint8(y))
	}
	
	
	sx := int64(x)
	sy := int64(y)
	checkint64(sx, sy)
	if (int64(int32(sx)) == sx || int64(int32(^sx)) == ^sx) && (int64(int32(sy)) == sy || int64(int32(^sy)) == ^sy) {
		checkint32(int32(sx), int32(sy))
	}
	if (int64(int16(sx)) == sx || int64(int16(^sx)) == ^sx) && (int64(int16(sy)) == sy || int64(int16(^sy)) == ^sy) {
		checkint16(int16(sx), int16(sy))
	}
	if (int64(int8(sx)) == sx || int64(int8(^sx)) == ^sx) && (int64(int8(sy)) == sy || int64(int8(^sy)) == ^sy) {
		checkint8(int8(sx), int8(sy))
	}
}

// Check result of x/y, x%y for various types.

func checkuint(x, y uint) {
	if y == 0 {
		divzerouint(x, y)
		modzerouint(x, y)
		return
	}
	q, r := udiv(uint64(x), uint64(y))
	q1 := x/y
	r1 := x%y
	if q1 != uint(q) {
		print("uint(", x, "/", y, ") = ", q1, ", want ", q, "\n")
	}
	if r1 != uint(r) {
		print("uint(", x, "%", y, ") = ", r1, ", want ", r, "\n")
	}
}

func checkuint64(x, y uint64) {
	if y == 0 {
		divzerouint64(x, y)
		modzerouint64(x, y)
		return
	}
	q, r := udiv(x, y)
	q1 := x/y
	r1 := x%y
	if q1 != q {
		print("uint64(", x, "/", y, ") = ", q1, ", want ", q, "\n")
	}
	if r1 != r {
		print("uint64(", x, "%", y, ") = ", r1, ", want ", r, "\n")
	}
}

func checkuint32(x, y uint32) {
	if y == 0 {
		divzerouint32(x, y)
		modzerouint32(x, y)
		return
	}
	q, r := udiv(uint64(x), uint64(y))
	q1 := x/y
	r1 := x%y
	if q1 != uint32(q) {
		print("uint32(", x, "/", y, ") = ", q1, ", want ", q, "\n")
	}
	if r1 != uint32(r) {
		print("uint32(", x, "%", y, ") = ", r1, ", want ", r, "\n")
	}
}

func checkuint16(x, y uint16) {
	if y == 0 {
		divzerouint16(x, y)
		modzerouint16(x, y)
		return
	}
	q, r := udiv(uint64(x), uint64(y))
	q1 := x/y
	r1 := x%y
	if q1 != uint16(q) {
		print("uint16(", x, "/", y, ") = ", q1, ", want ", q, "\n")
	}
	if r1 != uint16(r) {
		print("uint16(", x, "%", y, ") = ", r1, ", want ", r, "\n")
	}
}

func checkuint8(x, y uint8) {
	if y == 0 {
		divzerouint8(x, y)
		modzerouint8(x, y)
		return
	}
	q, r := udiv(uint64(x), uint64(y))
	q1 := x/y
	r1 := x%y
	if q1 != uint8(q) {
		print("uint8(", x, "/", y, ") = ", q1, ", want ", q, "\n")
	}
	if r1 != uint8(r) {
		print("uint8(", x, "%", y, ") = ", r1, ", want ", r, "\n")
	}
}

func checkint(x, y int) {
	if y == 0 {
		divzeroint(x, y)
		modzeroint(x, y)
		return
	}
	q, r := idiv(int64(x), int64(y))
	q1 := x/y
	r1 := x%y
	if q1 != int(q) {
		print("int(", x, "/", y, ") = ", q1, ", want ", q, "\n")
	}
	if r1 != int(r) {
		print("int(", x, "%", y, ") = ", r1, ", want ", r, "\n")
	}
}

func checkint64(x, y int64) {
	if y == 0 {
		divzeroint64(x, y)
		modzeroint64(x, y)
		return
	}
	q, r := idiv(x, y)
	q1 := x/y
	r1 := x%y
	if q1 != q {
		print("int64(", x, "/", y, ") = ", q1, ", want ", q, "\n")
	}
	if r1 != r {
		print("int64(", x, "%", y, ") = ", r1, ", want ", r, "\n")
	}
}

func checkint32(x, y int32) {
	if y == 0 {
		divzeroint32(x, y)
		modzeroint32(x, y)
		return
	}
	q, r := idiv(int64(x), int64(y))
	q1 := x/y
	r1 := x%y
	if q1 != int32(q) {
		print("int32(", x, "/", y, ") = ", q1, ", want ", q, "\n")
	}
	if r1 != int32(r) {
		print("int32(", x, "%", y, ") = ", r1, ", want ", r, "\n")
	}
}

func checkint16(x, y int16) {
	if y == 0 {
		divzeroint16(x, y)
		modzeroint16(x, y)
		return
	}
	q, r := idiv(int64(x), int64(y))
	q1 := x/y
	r1 := x%y
	if q1 != int16(q) {
		print("int16(", x, "/", y, ") = ", q1, ", want ", q, "\n")
	}
	if r1 != int16(r) {
		print("int16(", x, "%", y, ") = ", r1, ", want ", r, "\n")
	}
}

func checkint8(x, y int8) {
	if y == 0 {
		divzeroint8(x, y)
		modzeroint8(x, y)
		return
	}
	q, r := idiv(int64(x), int64(y))
	q1 := x/y
	r1 := x%y
	if q1 != int8(q) {
		print("int8(", x, "/", y, ") = ", q1, ", want ", q, "\n")
	}
	if r1 != int8(r) {
		print("int8(", x, "%", y, ") = ", r1, ", want ", r, "\n")
	}
}

func divzerouint(x, y uint) uint {
	defer checkudivzero("uint", uint64(x))
	return x / y
}

func divzerouint64(x, y uint64) uint64 {
	defer checkudivzero("uint64", uint64(x))
	return x / y
}

func divzerouint32(x, y uint32) uint32 {
	defer checkudivzero("uint32", uint64(x))
	return x / y
}

func divzerouint16(x, y uint16) uint16 {
	defer checkudivzero("uint16", uint64(x))
	return x / y
}

func divzerouint8(x, y uint8) uint8 {
	defer checkudivzero("uint8", uint64(x))
	return x / y
}

func checkudivzero(typ string, x uint64) {
	if recover() == nil {
		print(typ, "(", x, " / 0) did not panic")
	}
}

func divzeroint(x, y int) int {
	defer checkdivzero("int", int64(x))
	return x / y
}

func divzeroint64(x, y int64) int64 {
	defer checkdivzero("int64", int64(x))
	return x / y
}

func divzeroint32(x, y int32) int32 {
	defer checkdivzero("int32", int64(x))
	return x / y
}

func divzeroint16(x, y int16) int16 {
	defer checkdivzero("int16", int64(x))
	return x / y
}

func divzeroint8(x, y int8) int8 {
	defer checkdivzero("int8", int64(x))
	return x / y
}

func checkdivzero(typ string, x int64) {
	if recover() == nil {
		print(typ, "(", x, " / 0) did not panic")
	}
}

func modzerouint(x, y uint) uint {
	defer checkumodzero("uint", uint64(x))
	return x % y
}

func modzerouint64(x, y uint64) uint64 {
	defer checkumodzero("uint64", uint64(x))
	return x % y
}

func modzerouint32(x, y uint32) uint32 {
	defer checkumodzero("uint32", uint64(x))
	return x % y
}

func modzerouint16(x, y uint16) uint16 {
	defer checkumodzero("uint16", uint64(x))
	return x % y
}

func modzerouint8(x, y uint8) uint8 {
	defer checkumodzero("uint8", uint64(x))
	return x % y
}

func checkumodzero(typ string, x uint64) {
	if recover() == nil {
		print(typ, "(", x, " % 0) did not panic")
	}
}

func modzeroint(x, y int) int {
	defer checkmodzero("int", int64(x))
	return x % y
}

func modzeroint64(x, y int64) int64 {
	defer checkmodzero("int64", int64(x))
	return x % y
}

func modzeroint32(x, y int32) int32 {
	defer checkmodzero("int32", int64(x))
	return x % y
}

func modzeroint16(x, y int16) int16 {
	defer checkmodzero("int16", int64(x))
	return x % y
}

func modzeroint8(x, y int8) int8 {
	defer checkmodzero("int8", int64(x))
	return x % y
}

func checkmodzero(typ string, x int64) {
	if recover() == nil {
		print(typ, "(", x, " % 0) did not panic")
	}
}

// unsigned divide and mod using shift and subtract.
func udiv(x, y uint64) (q, r uint64) {
	sh := 0
	for y+y > y && y+y <= x {
		sh++
		y <<= 1
	}
	for ; sh >= 0; sh-- {
		q <<= 1
		if x >= y {
			x -= y
			q |= 1
		}
		y >>= 1
	}
	return q, x	
}

// signed divide and mod: do unsigned and adjust signs.
func idiv(x, y int64) (q, r int64) {
	// special case for minint / -1 = minint
	if x-1 > x && y == -1 {
		return x, 0
	}
	ux := uint64(x)
	uy := uint64(y)
	if x < 0 {
		ux = -ux
	}
	if y < 0 {
		uy = -uy
	}
	uq, ur := udiv(ux, uy)
	q = int64(uq)
	r = int64(ur)
	if x < 0 {
		r = -r
	}
	if (x < 0) != (y < 0) {
		q = -q
	}
	return q, r
}
