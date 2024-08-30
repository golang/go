// errorcheck -0 -d=ssa/prove/debug=1

//go:build amd64

package main

import "math"

func f0a(a []int) int {
	x := 0
	for i := range a { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		x += a[i] // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return x
}

func f0b(a []int) int {
	x := 0
	for i := range a { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		b := a[i:] // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		x += b[0]
	}
	return x
}

func f0c(a []int) int {
	x := 0
	for i := range a { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		b := a[:i+1] // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		x += b[0]    // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return x
}

func f1(a []int) int {
	x := 0
	for _, i := range a { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		x += i
	}
	return x
}

func f2(a []int) int {
	x := 0
	for i := 1; i < len(a); i++ { // ERROR "Induction variable: limits \[1,\?\), increment 1$"
		x += a[i] // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return x
}

func f4(a [10]int) int {
	x := 0
	for i := 0; i < len(a); i += 2 { // ERROR "Induction variable: limits \[0,8\], increment 2$"
		x += a[i] // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return x
}

func f5(a [10]int) int {
	x := 0
	for i := -10; i < len(a); i += 2 { // ERROR "Induction variable: limits \[-10,8\], increment 2$"
		x += a[i+10]
	}
	return x
}

func f5_int32(a [10]int) int {
	x := 0
	for i := int32(-10); i < int32(len(a)); i += 2 { // ERROR "Induction variable: limits \[-10,8\], increment 2$"
		x += a[i+10]
	}
	return x
}

func f5_int16(a [10]int) int {
	x := 0
	for i := int16(-10); i < int16(len(a)); i += 2 { // ERROR "Induction variable: limits \[-10,8\], increment 2$"
		x += a[i+10]
	}
	return x
}

func f5_int8(a [10]int) int {
	x := 0
	for i := int8(-10); i < int8(len(a)); i += 2 { // ERROR "Induction variable: limits \[-10,8\], increment 2$"
		x += a[i+10]
	}
	return x
}

func f6(a []int) {
	for i := range a { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		b := a[0:i] // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		f6(b)
	}
}

func g0a(a string) int {
	x := 0
	for i := 0; i < len(a); i++ { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		x += int(a[i]) // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return x
}

func g0b(a string) int {
	x := 0
	for i := 0; len(a) > i; i++ { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		x += int(a[i]) // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return x
}

func g0c(a string) int {
	x := 0
	for i := len(a); i > 0; i-- { // ERROR "Induction variable: limits \(0,\?\], increment 1$"
		x += int(a[i-1]) // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return x
}

func g0d(a string) int {
	x := 0
	for i := len(a); 0 < i; i-- { // ERROR "Induction variable: limits \(0,\?\], increment 1$"
		x += int(a[i-1]) // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return x
}

func g0e(a string) int {
	x := 0
	for i := len(a) - 1; i >= 0; i-- { // ERROR "Induction variable: limits \[0,\?\], increment 1$"
		x += int(a[i]) // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return x
}

func g0f(a string) int {
	x := 0
	for i := len(a) - 1; 0 <= i; i-- { // ERROR "Induction variable: limits \[0,\?\], increment 1$"
		x += int(a[i]) // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return x
}

func g1() int {
	a := "evenlength"
	x := 0
	for i := 0; i < len(a); i += 2 { // ERROR "Induction variable: limits \[0,8\], increment 2$"
		x += int(a[i]) // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return x
}

func g2() int {
	a := "evenlength"
	x := 0
	for i := 0; i < len(a); i += 2 { // ERROR "Induction variable: limits \[0,8\], increment 2$"
		j := i
		if a[i] == 'e' { // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
			j = j + 1
		}
		x += int(a[j])
	}
	return x
}

func g3a() {
	a := "this string has length 25"
	for i := 0; i < len(a); i += 5 { // ERROR "Induction variable: limits \[0,20\], increment 5$"
		useString(a[i:])   // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useString(a[:i+3]) // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useString(a[:i+5]) // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useString(a[:i+6])
	}
}

func g3b(a string) {
	for i := 0; i < len(a); i++ { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		useString(a[i+1:]) // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
	}
}

func g3c(a string) {
	for i := 0; i < len(a); i++ { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		useString(a[:i+1]) // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
	}
}

func h1(a []byte) {
	c := a[:128]
	for i := range c { // ERROR "Induction variable: limits \[0,128\), increment 1$"
		c[i] = byte(i) // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
}

func h2(a []byte) {
	for i := range a[:128] { // ERROR "Induction variable: limits \[0,128\), increment 1$"
		a[i] = byte(i)
	}
}

func k0(a [100]int) [100]int {
	for i := 10; i < 90; i++ { // ERROR "Induction variable: limits \[10,90\), increment 1$"
		if a[0] == 0xdeadbeef {
			// This is a trick to prohibit sccp to optimize out the following out of bound check
			continue
		}
		a[i-11] = i
		a[i-10] = i // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
		a[i-5] = i  // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
		a[i] = i    // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
		a[i+5] = i  // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
		a[i+10] = i // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
		a[i+11] = i
	}
	return a
}

func k1(a [100]int) [100]int {
	for i := 10; i < 90; i++ { // ERROR "Induction variable: limits \[10,90\), increment 1$"
		if a[0] == 0xdeadbeef {
			// This is a trick to prohibit sccp to optimize out the following out of bound check
			continue
		}
		useSlice(a[:i-11])
		useSlice(a[:i-10]) // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useSlice(a[:i-5])  // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useSlice(a[:i])    // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useSlice(a[:i+5])  // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useSlice(a[:i+10]) // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useSlice(a[:i+11]) // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useSlice(a[:i+12])

	}
	return a
}

func k2(a [100]int) [100]int {
	for i := 10; i < 90; i++ { // ERROR "Induction variable: limits \[10,90\), increment 1$"
		if a[0] == 0xdeadbeef {
			// This is a trick to prohibit sccp to optimize out the following out of bound check
			continue
		}
		useSlice(a[i-11:])
		useSlice(a[i-10:]) // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useSlice(a[i-5:])  // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useSlice(a[i:])    // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useSlice(a[i+5:])  // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useSlice(a[i+10:]) // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useSlice(a[i+11:]) // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
		useSlice(a[i+12:])
	}
	return a
}

func k3(a [100]int) [100]int {
	for i := -10; i < 90; i++ { // ERROR "Induction variable: limits \[-10,90\), increment 1$"
		if a[0] == 0xdeadbeef {
			// This is a trick to prohibit sccp to optimize out the following out of bound check
			continue
		}
		a[i+9] = i
		a[i+10] = i // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
		a[i+11] = i
	}
	return a
}

func k3neg(a [100]int) [100]int {
	for i := 89; i > -11; i-- { // ERROR "Induction variable: limits \(-11,89\], increment 1$"
		if a[0] == 0xdeadbeef {
			// This is a trick to prohibit sccp to optimize out the following out of bound check
			continue
		}
		a[i+9] = i
		a[i+10] = i // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
		a[i+11] = i
	}
	return a
}

func k3neg2(a [100]int) [100]int {
	for i := 89; i >= -10; i-- { // ERROR "Induction variable: limits \[-10,89\], increment 1$"
		if a[0] == 0xdeadbeef {
			// This is a trick to prohibit sccp to optimize out the following out of bound check
			continue
		}
		a[i+9] = i
		a[i+10] = i // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
		a[i+11] = i
	}
	return a
}

func k4(a [100]int) [100]int {
	// Note: can't use (-1)<<63 here, because i-min doesn't get rewritten to i+(-min),
	// and it isn't worth adding that special case to prove.
	min := (-1)<<63 + 1
	for i := min; i < min+50; i++ { // ERROR "Induction variable: limits \[-9223372036854775807,-9223372036854775757\), increment 1$"
		a[i-min] = i // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return a
}

func k5(a [100]int) [100]int {
	max := (1 << 63) - 1
	for i := max - 50; i < max; i++ { // ERROR "Induction variable: limits \[9223372036854775757,9223372036854775807\), increment 1$"
		a[i-max+50] = i   // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
		a[i-(max-70)] = i // ERROR "(\([0-9]+\) )?Proved IsInBounds$"
	}
	return a
}

func d1(a [100]int) [100]int {
	for i := 0; i < 100; i++ { // ERROR "Induction variable: limits \[0,100\), increment 1$"
		for j := 0; j < i; j++ { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
			a[j] = 0   // ERROR "Proved IsInBounds$"
			a[j+1] = 0 // ERROR "Proved IsInBounds$"
			a[j+2] = 0
		}
	}
	return a
}

func d2(a [100]int) [100]int {
	for i := 0; i < 100; i++ { // ERROR "Induction variable: limits \[0,100\), increment 1$"
		for j := 0; i > j; j++ { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
			a[j] = 0   // ERROR "Proved IsInBounds$"
			a[j+1] = 0 // ERROR "Proved IsInBounds$"
			a[j+2] = 0
		}
	}
	return a
}

func d3(a [100]int) [100]int {
	for i := 0; i <= 99; i++ { // ERROR "Induction variable: limits \[0,99\], increment 1$"
		for j := 0; j <= i-1; j++ {
			a[j] = 0
			a[j+1] = 0 // ERROR "Proved IsInBounds$"
			a[j+2] = 0
		}
	}
	return a
}

func d4() {
	for i := int64(math.MaxInt64 - 9); i < math.MaxInt64-2; i += 4 { // ERROR "Induction variable: limits \[9223372036854775798,9223372036854775802\], increment 4$"
		useString("foo")
	}
	for i := int64(math.MaxInt64 - 8); i < math.MaxInt64-2; i += 4 { // ERROR "Induction variable: limits \[9223372036854775799,9223372036854775803\], increment 4$"
		useString("foo")
	}
	for i := int64(math.MaxInt64 - 7); i < math.MaxInt64-2; i += 4 {
		useString("foo")
	}
	for i := int64(math.MaxInt64 - 6); i < math.MaxInt64-2; i += 4 { // ERROR "Induction variable: limits \[9223372036854775801,9223372036854775801\], increment 4$"
		useString("foo")
	}
	for i := int64(math.MaxInt64 - 9); i <= math.MaxInt64-2; i += 4 { // ERROR "Induction variable: limits \[9223372036854775798,9223372036854775802\], increment 4$"
		useString("foo")
	}
	for i := int64(math.MaxInt64 - 8); i <= math.MaxInt64-2; i += 4 { // ERROR "Induction variable: limits \[9223372036854775799,9223372036854775803\], increment 4$"
		useString("foo")
	}
	for i := int64(math.MaxInt64 - 7); i <= math.MaxInt64-2; i += 4 {
		useString("foo")
	}
	for i := int64(math.MaxInt64 - 6); i <= math.MaxInt64-2; i += 4 {
		useString("foo")
	}
}

func d5() {
	for i := int64(math.MinInt64 + 9); i > math.MinInt64+2; i -= 4 { // ERROR "Induction variable: limits \[-9223372036854775803,-9223372036854775799\], increment 4"
		useString("foo")
	}
	for i := int64(math.MinInt64 + 8); i > math.MinInt64+2; i -= 4 { // ERROR "Induction variable: limits \[-9223372036854775804,-9223372036854775800\], increment 4"
		useString("foo")
	}
	for i := int64(math.MinInt64 + 7); i > math.MinInt64+2; i -= 4 {
		useString("foo")
	}
	for i := int64(math.MinInt64 + 6); i > math.MinInt64+2; i -= 4 { // ERROR "Induction variable: limits \[-9223372036854775802,-9223372036854775802\], increment 4"
		useString("foo")
	}
	for i := int64(math.MinInt64 + 9); i >= math.MinInt64+2; i -= 4 { // ERROR "Induction variable: limits \[-9223372036854775803,-9223372036854775799\], increment 4"
		useString("foo")
	}
	for i := int64(math.MinInt64 + 8); i >= math.MinInt64+2; i -= 4 { // ERROR "Induction variable: limits \[-9223372036854775804,-9223372036854775800\], increment 4"
		useString("foo")
	}
	for i := int64(math.MinInt64 + 7); i >= math.MinInt64+2; i -= 4 {
		useString("foo")
	}
	for i := int64(math.MinInt64 + 6); i >= math.MinInt64+2; i -= 4 {
		useString("foo")
	}
}

func bce1() {
	// tests overflow of max-min
	a := int64(9223372036854774057)
	b := int64(-1547)
	z := int64(1337)

	if a%z == b%z {
		panic("invalid test: modulos should differ")
	}

	for i := b; i < a; i += z { // ERROR "Induction variable: limits \[-1547,9223372036854772720\], increment 1337"
		useString("foobar")
	}
}

func nobce2(a string) {
	for i := int64(0); i < int64(len(a)); i++ { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		useString(a[i:]) // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
	}
	for i := int64(0); i < int64(len(a))-31337; i++ { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		useString(a[i:]) // ERROR "(\([0-9]+\) )?Proved IsSliceInBounds$"
	}
	for i := int64(0); i < int64(len(a))+int64(-1<<63); i++ { // ERROR "Induction variable: limits \[0,\?\), increment 1$" "Disproved Less64"
		useString(a[i:])
	}
	j := int64(len(a)) - 123
	for i := int64(0); i < j+123+int64(-1<<63); i++ { // ERROR "Induction variable: limits \[0,\?\), increment 1$" "Disproved Less64"
		useString(a[i:])
	}
	for i := int64(0); i < j+122+int64(-1<<63); i++ { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		// len(a)-123+122+MinInt overflows when len(a) == 0, so a bound check is needed here
		useString(a[i:])
	}
}

func nobce3(a [100]int64) [100]int64 {
	min := int64((-1) << 63)
	max := int64((1 << 63) - 1)
	for i := min; i < max; i++ { // ERROR "Induction variable: limits \[-9223372036854775808,9223372036854775807\), increment 1$"
	}
	return a
}

func issue26116a(a []int) {
	// There is no induction variable here. The comparison is in the wrong direction.
	for i := 3; i > 6; i++ {
		a[i] = 0
	}
	for i := 7; i < 3; i-- {
		a[i] = 1
	}
}

func stride1(x *[7]int) int {
	s := 0
	for i := 0; i <= 8; i += 3 { // ERROR "Induction variable: limits \[0,6\], increment 3"
		s += x[i] // ERROR "Proved IsInBounds"
	}
	return s
}

func stride2(x *[7]int) int {
	s := 0
	for i := 0; i < 9; i += 3 { // ERROR "Induction variable: limits \[0,6\], increment 3"
		s += x[i] // ERROR "Proved IsInBounds"
	}
	return s
}

//go:noinline
func useString(a string) {
}

//go:noinline
func useSlice(a []int) {
}

func main() {
}
