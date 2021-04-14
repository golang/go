// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big_test

import (
	"fmt"
	"math"
	"math/big"
)

func ExampleFloat_Add() {
	// Operate on numbers of different precision.
	var x, y, z big.Float
	x.SetInt64(1000)          // x is automatically set to 64bit precision
	y.SetFloat64(2.718281828) // y is automatically set to 53bit precision
	z.SetPrec(32)
	z.Add(&x, &y)
	fmt.Printf("x = %.10g (%s, prec = %d, acc = %s)\n", &x, x.Text('p', 0), x.Prec(), x.Acc())
	fmt.Printf("y = %.10g (%s, prec = %d, acc = %s)\n", &y, y.Text('p', 0), y.Prec(), y.Acc())
	fmt.Printf("z = %.10g (%s, prec = %d, acc = %s)\n", &z, z.Text('p', 0), z.Prec(), z.Acc())
	// Output:
	// x = 1000 (0x.fap+10, prec = 64, acc = Exact)
	// y = 2.718281828 (0x.adf85458248cd8p+2, prec = 53, acc = Exact)
	// z = 1002.718282 (0x.faadf854p+10, prec = 32, acc = Below)
}

func ExampleFloat_shift() {
	// Implement Float "shift" by modifying the (binary) exponents directly.
	for s := -5; s <= 5; s++ {
		x := big.NewFloat(0.5)
		x.SetMantExp(x, x.MantExp(nil)+s) // shift x by s
		fmt.Println(x)
	}
	// Output:
	// 0.015625
	// 0.03125
	// 0.0625
	// 0.125
	// 0.25
	// 0.5
	// 1
	// 2
	// 4
	// 8
	// 16
}

func ExampleFloat_Cmp() {
	inf := math.Inf(1)
	zero := 0.0

	operands := []float64{-inf, -1.2, -zero, 0, +1.2, +inf}

	fmt.Println("   x     y  cmp")
	fmt.Println("---------------")
	for _, x64 := range operands {
		x := big.NewFloat(x64)
		for _, y64 := range operands {
			y := big.NewFloat(y64)
			fmt.Printf("%4g  %4g  %3d\n", x, y, x.Cmp(y))
		}
		fmt.Println()
	}

	// Output:
	//    x     y  cmp
	// ---------------
	// -Inf  -Inf    0
	// -Inf  -1.2   -1
	// -Inf    -0   -1
	// -Inf     0   -1
	// -Inf   1.2   -1
	// -Inf  +Inf   -1
	//
	// -1.2  -Inf    1
	// -1.2  -1.2    0
	// -1.2    -0   -1
	// -1.2     0   -1
	// -1.2   1.2   -1
	// -1.2  +Inf   -1
	//
	//   -0  -Inf    1
	//   -0  -1.2    1
	//   -0    -0    0
	//   -0     0    0
	//   -0   1.2   -1
	//   -0  +Inf   -1
	//
	//    0  -Inf    1
	//    0  -1.2    1
	//    0    -0    0
	//    0     0    0
	//    0   1.2   -1
	//    0  +Inf   -1
	//
	//  1.2  -Inf    1
	//  1.2  -1.2    1
	//  1.2    -0    1
	//  1.2     0    1
	//  1.2   1.2    0
	//  1.2  +Inf   -1
	//
	// +Inf  -Inf    1
	// +Inf  -1.2    1
	// +Inf    -0    1
	// +Inf     0    1
	// +Inf   1.2    1
	// +Inf  +Inf    0
}

func ExampleRoundingMode() {
	operands := []float64{2.6, 2.5, 2.1, -2.1, -2.5, -2.6}

	fmt.Print("   x")
	for mode := big.ToNearestEven; mode <= big.ToPositiveInf; mode++ {
		fmt.Printf("  %s", mode)
	}
	fmt.Println()

	for _, f64 := range operands {
		fmt.Printf("%4g", f64)
		for mode := big.ToNearestEven; mode <= big.ToPositiveInf; mode++ {
			// sample operands above require 2 bits to represent mantissa
			// set binary precision to 2 to round them to integer values
			f := new(big.Float).SetPrec(2).SetMode(mode).SetFloat64(f64)
			fmt.Printf("  %*g", len(mode.String()), f)
		}
		fmt.Println()
	}

	// Output:
	//    x  ToNearestEven  ToNearestAway  ToZero  AwayFromZero  ToNegativeInf  ToPositiveInf
	//  2.6              3              3       2             3              2              3
	//  2.5              2              3       2             3              2              3
	//  2.1              2              2       2             3              2              3
	// -2.1             -2             -2      -2            -3             -3             -2
	// -2.5             -2             -3      -2            -3             -3             -2
	// -2.6             -3             -3      -2            -3             -3             -2
}
