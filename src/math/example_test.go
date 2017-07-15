// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math_test

import (
	"fmt"
	"math"
)

func ExampleSqrt() {
	const (
		a = 3
		b = 4
	)
	c := math.Sqrt(a*a + b*b)
	fmt.Printf("%.1f", c)
	// Output: 5.0
}
