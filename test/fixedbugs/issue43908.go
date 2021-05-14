// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify exact constant evaluation independent of
// (mathematically equivalent) expression form.

package main

import "fmt"

const ulp1 = imag(1i + 2i / 3 - 5i / 3)
const ulp2 = imag(1i + complex(0, 2) / 3 - 5i / 3)

func main() {
	if ulp1 != ulp2 {
		panic(fmt.Sprintf("%g != %g\n", ulp1, ulp2))
	}
}
