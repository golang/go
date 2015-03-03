// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big_test

import (
	"fmt"
	"math/big"
)

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
