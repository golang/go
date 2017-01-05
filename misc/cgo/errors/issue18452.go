// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 18452: show pos info in undefined name errors

package p

import (
	"C"
	"fmt"
)

func a() {
	fmt.Println("Hello, world!")
	C.function_that_does_not_exist() // line 16
	C.pi                             // line 17
}
