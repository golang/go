// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmplx_test

import (
	"fmt"
	"math"
	"math/cmplx"
)

func ExampleAbs() {
	fmt.Printf("%.1f", cmplx.Abs(3+4i))
	// Output: 5.0
}

// ExampleExp computes Euler's identity.
func ExampleExp() {
	fmt.Printf("%.1f", cmplx.Exp(1i*math.Pi)+1)
	// Output: (0.0+0.0i)
}

func ExamplePolar() {
	r, theta := cmplx.Polar(2i)
	fmt.Printf("r: %.1f, θ: %.1f*π", r, theta/math.Pi)
	// Output: r: 2.0, θ: 0.5*π
}
