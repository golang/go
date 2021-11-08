// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Compile-time constants, even if they cannot be represented
// accurately, should remain the same in operations that don't
// affect their values.

package main

import "fmt"

func main() {
	const x = 0.01
	const xi = 0.01i
	const xc = complex(0, x)

	if imag(xi) != x {
		fmt.Printf("FAILED: %g != %g\n", imag(xi), x)
	}

	if xi != complex(0, x) {
		fmt.Printf("FAILED: %g != %g\n", xi, complex(0, x))
	}
}
