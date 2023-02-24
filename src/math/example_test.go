// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math_test

import (
	"fmt"
	"math"
)

func ExampleAcos() {
	fmt.Printf("%.2f", math.Acos(1))
	// Output: 0.00
}

func ExampleAcosh() {
	fmt.Printf("%.2f", math.Acosh(1))
	// Output: 0.00
}

func ExampleAsin() {
	fmt.Printf("%.2f", math.Asin(0))
	// Output: 0.00
}

func ExampleAsinh() {
	fmt.Printf("%.2f", math.Asinh(0))
	// Output: 0.00
}

func ExampleAtan() {
	fmt.Printf("%.2f", math.Atan(0))
	// Output: 0.00
}

func ExampleAtan2() {
	fmt.Printf("%.2f", math.Atan2(0, 0))
	// Output: 0.00
}

func ExampleAtanh() {
	fmt.Printf("%.2f", math.Atanh(0))
	// Output: 0.00
}

func ExampleCopysign() {
	fmt.Printf("%.2f", math.Copysign(3.2, -1))
	// Output: -3.20
}

func ExampleCos() {
	fmt.Printf("%.2f", math.Cos(math.Pi/2))
	// Output: 0.00
}

func ExampleCosh() {
	fmt.Printf("%.2f", math.Cosh(0))
	// Output: 1.00
}

func ExampleSin() {
	fmt.Printf("%.2f", math.Sin(math.Pi))
	// Output: 0.00
}

func ExampleSincos() {
	sin, cos := math.Sincos(0)
	fmt.Printf("%.2f, %.2f", sin, cos)
	// Output: 0.00, 1.00
}

func ExampleSinh() {
	fmt.Printf("%.2f", math.Sinh(0))
	// Output: 0.00
}

func ExampleTan() {
	fmt.Printf("%.2f", math.Tan(0))
	// Output: 0.00
}

func ExampleTanh() {
	fmt.Printf("%.2f", math.Tanh(0))
	// Output: 0.00
}

func ExampleSqrt() {
	const (
		a = 3
		b = 4
	)
	c := math.Sqrt(a*a + b*b)
	fmt.Printf("%.1f", c)
	// Output: 5.0
}

func ExampleCeil() {
	c := math.Ceil(1.49)
	fmt.Printf("%.1f", c)
	// Output: 2.0
}

func ExampleFloor() {
	c := math.Floor(1.51)
	fmt.Printf("%.1f", c)
	// Output: 1.0
}

func ExamplePow() {
	c := math.Pow(2, 3)
	fmt.Printf("%.1f", c)
	// Output: 8.0
}

func ExamplePow10() {
	c := math.Pow10(2)
	fmt.Printf("%.1f", c)
	// Output: 100.0
}

func ExampleRound() {
	p := math.Round(10.5)
	fmt.Printf("%.1f\n", p)

	n := math.Round(-10.5)
	fmt.Printf("%.1f\n", n)
	// Output:
	// 11.0
	// -11.0
}

func ExampleRoundToEven() {
	u := math.RoundToEven(11.5)
	fmt.Printf("%.1f\n", u)

	d := math.RoundToEven(12.5)
	fmt.Printf("%.1f\n", d)
	// Output:
	// 12.0
	// 12.0
}

func ExampleLog() {
	x := math.Log(1)
	fmt.Printf("%.1f\n", x)

	y := math.Log(2.7183)
	fmt.Printf("%.1f\n", y)
	// Output:
	// 0.0
	// 1.0
}

func ExampleLog2() {
	fmt.Printf("%.1f", math.Log2(256))
	// Output: 8.0
}

func ExampleLog10() {
	fmt.Printf("%.1f", math.Log10(100))
	// Output: 2.0
}

func ExampleRemainder() {
	fmt.Printf("%.1f", math.Remainder(100, 30))
	// Output: 10.0
}

func ExampleMod() {
	c := math.Mod(7, 4)
	fmt.Printf("%.1f", c)
	// Output: 3.0
}

func ExampleAbs() {
	x := math.Abs(-2)
	fmt.Printf("%.1f\n", x)

	y := math.Abs(2)
	fmt.Printf("%.1f\n", y)
	// Output:
	// 2.0
	// 2.0
}
func ExampleDim() {
	fmt.Printf("%.2f\n", math.Dim(4, -2))
	fmt.Printf("%.2f\n", math.Dim(-4, 2))
	// Output:
	// 6.00
	// 0.00
}

func ExampleExp() {
	fmt.Printf("%.2f\n", math.Exp(1))
	fmt.Printf("%.2f\n", math.Exp(2))
	fmt.Printf("%.2f\n", math.Exp(-1))
	// Output:
	// 2.72
	// 7.39
	// 0.37
}

func ExampleExp2() {
	fmt.Printf("%.2f\n", math.Exp2(1))
	fmt.Printf("%.2f\n", math.Exp2(-3))
	// Output:
	// 2.00
	// 0.12
}

func ExampleExpm1() {
	fmt.Printf("%.6f\n", math.Expm1(0.01))
	fmt.Printf("%.6f\n", math.Expm1(-1))
	// Output:
	// 0.010050
	// -0.632121
}

func ExampleTrunc() {
	fmt.Printf("%.2f\n", math.Trunc(math.Pi))
	fmt.Printf("%.2f\n", math.Trunc(-1.2345))
	// Output:
	// 3.00
	// -1.00
}

func ExampleCbrt() {
	fmt.Printf("%.2f\n", math.Cbrt(8))
	fmt.Printf("%.2f\n", math.Cbrt(27))
	// Output:
	// 2.00
	// 3.00
}

func ExampleModf() {
	int, frac := math.Modf(3.14)
	fmt.Printf("%.2f, %.2f\n", int, frac)

	int, frac = math.Modf(-2.71)
	fmt.Printf("%.2f, %.2f\n", int, frac)
	// Output:
	// 3.00, 0.14
	// -2.00, -0.71
}
