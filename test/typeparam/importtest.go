// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file checks that basic importing works in -G mode.

package p

import "fmt"
import "math"

func f(x float64) {
	fmt.Println(math.Sin(x))
}
