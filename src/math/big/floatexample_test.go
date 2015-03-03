// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big_test

import (
	"fmt"
	"math/big"
)

// TODO(gri) add more examples

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
	var x big.Float
	for s := -5; s <= 5; s++ {
		x.SetFloat64(0.5)
		x.SetMantExp(&x, x.MantExp(nil)+s) // shift x by s
		fmt.Println(&x)
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
