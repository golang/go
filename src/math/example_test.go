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
