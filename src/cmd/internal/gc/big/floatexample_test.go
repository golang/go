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
	// Operating on numbers of different precision.
	var x, y, z big.Float
	x.SetInt64(1000)          // x is automatically set to 64bit precision
	y.SetFloat64(2.718281828) // y is automatically set to 53bit precision
	z.SetPrec(32)
	z.Add(&x, &y)
	fmt.Printf("x = %s (%s, prec = %d, acc = %s)\n", &x, x.Format('p', 0), x.Prec(), x.Acc())
	fmt.Printf("y = %s (%s, prec = %d, acc = %s)\n", &y, y.Format('p', 0), y.Prec(), y.Acc())
	fmt.Printf("z = %s (%s, prec = %d, acc = %s)\n", &z, z.Format('p', 0), z.Prec(), z.Acc())
	// Output:
	// x = 1000 (0x.fap10, prec = 64, acc = Exact)
	// y = 2.718281828 (0x.adf85458248cd8p2, prec = 53, acc = Exact)
	// z = 1002.718282 (0x.faadf854p10, prec = 32, acc = Below)
}

func Example_Shift() {
	// Implementing Float "shift" by modifying the (binary) exponents directly.
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
	nan := math.NaN()

	operands := []float64{-inf, -1.2, -zero, 0, +1.2, +inf, nan}

	fmt.Println("   x     y   cmp   eql  neq  lss  leq  gtr  geq")
	fmt.Println("-----------------------------------------------")
	for _, x64 := range operands {
		x := big.NewFloat(x64)
		for _, y64 := range operands {
			y := big.NewFloat(y64)
			t := x.Cmp(y)
			fmt.Printf(
				"%4s  %4s  %5s   %c    %c    %c    %c    %c    %c\n",
				x, y, t.Acc(),
				mark(t.Eql()), mark(t.Neq()), mark(t.Lss()), mark(t.Leq()), mark(t.Gtr()), mark(t.Geq()))
		}
		fmt.Println()
	}

	// Output:
	//    x     y   cmp   eql  neq  lss  leq  gtr  geq
	// -----------------------------------------------
	// -Inf  -Inf  Exact   ●    ○    ○    ●    ○    ●
	// -Inf  -1.2  Below   ○    ●    ●    ●    ○    ○
	// -Inf    -0  Below   ○    ●    ●    ●    ○    ○
	// -Inf     0  Below   ○    ●    ●    ●    ○    ○
	// -Inf   1.2  Below   ○    ●    ●    ●    ○    ○
	// -Inf  +Inf  Below   ○    ●    ●    ●    ○    ○
	// -Inf   NaN  Undef   ○    ●    ○    ○    ○    ○
	//
	// -1.2  -Inf  Above   ○    ●    ○    ○    ●    ●
	// -1.2  -1.2  Exact   ●    ○    ○    ●    ○    ●
	// -1.2    -0  Below   ○    ●    ●    ●    ○    ○
	// -1.2     0  Below   ○    ●    ●    ●    ○    ○
	// -1.2   1.2  Below   ○    ●    ●    ●    ○    ○
	// -1.2  +Inf  Below   ○    ●    ●    ●    ○    ○
	// -1.2   NaN  Undef   ○    ●    ○    ○    ○    ○
	//
	//   -0  -Inf  Above   ○    ●    ○    ○    ●    ●
	//   -0  -1.2  Above   ○    ●    ○    ○    ●    ●
	//   -0    -0  Exact   ●    ○    ○    ●    ○    ●
	//   -0     0  Exact   ●    ○    ○    ●    ○    ●
	//   -0   1.2  Below   ○    ●    ●    ●    ○    ○
	//   -0  +Inf  Below   ○    ●    ●    ●    ○    ○
	//   -0   NaN  Undef   ○    ●    ○    ○    ○    ○
	//
	//    0  -Inf  Above   ○    ●    ○    ○    ●    ●
	//    0  -1.2  Above   ○    ●    ○    ○    ●    ●
	//    0    -0  Exact   ●    ○    ○    ●    ○    ●
	//    0     0  Exact   ●    ○    ○    ●    ○    ●
	//    0   1.2  Below   ○    ●    ●    ●    ○    ○
	//    0  +Inf  Below   ○    ●    ●    ●    ○    ○
	//    0   NaN  Undef   ○    ●    ○    ○    ○    ○
	//
	//  1.2  -Inf  Above   ○    ●    ○    ○    ●    ●
	//  1.2  -1.2  Above   ○    ●    ○    ○    ●    ●
	//  1.2    -0  Above   ○    ●    ○    ○    ●    ●
	//  1.2     0  Above   ○    ●    ○    ○    ●    ●
	//  1.2   1.2  Exact   ●    ○    ○    ●    ○    ●
	//  1.2  +Inf  Below   ○    ●    ●    ●    ○    ○
	//  1.2   NaN  Undef   ○    ●    ○    ○    ○    ○
	//
	// +Inf  -Inf  Above   ○    ●    ○    ○    ●    ●
	// +Inf  -1.2  Above   ○    ●    ○    ○    ●    ●
	// +Inf    -0  Above   ○    ●    ○    ○    ●    ●
	// +Inf     0  Above   ○    ●    ○    ○    ●    ●
	// +Inf   1.2  Above   ○    ●    ○    ○    ●    ●
	// +Inf  +Inf  Exact   ●    ○    ○    ●    ○    ●
	// +Inf   NaN  Undef   ○    ●    ○    ○    ○    ○
	//
	//  NaN  -Inf  Undef   ○    ●    ○    ○    ○    ○
	//  NaN  -1.2  Undef   ○    ●    ○    ○    ○    ○
	//  NaN    -0  Undef   ○    ●    ○    ○    ○    ○
	//  NaN     0  Undef   ○    ●    ○    ○    ○    ○
	//  NaN   1.2  Undef   ○    ●    ○    ○    ○    ○
	//  NaN  +Inf  Undef   ○    ●    ○    ○    ○    ○
	//  NaN   NaN  Undef   ○    ●    ○    ○    ○    ○
}

func mark(p bool) rune {
	if p {
		return '●'
	}
	return '○'
}
