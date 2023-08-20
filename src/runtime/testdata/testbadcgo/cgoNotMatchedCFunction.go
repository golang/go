// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testbadcgo

// no matched C function.

/*
#cgo noescape noMatchedCFunction
*/
import "C"

import (
	"fmt"
)

func init() {
	register("CgoNoMatchedCFunction", CgoNotMatchedCFunction)
}

func CgoNotMatchedCFunction() {
	fmt.Println("OK")
}
