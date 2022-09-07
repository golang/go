// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importdecl0

import "math"
import m "math"

import . "testing" // declares T in file scope
import . /* ERROR .unsafe. imported but not used */ "unsafe"
import . "fmt"     // declares Println in file scope

import (
	"" /* ERROR invalid import path */
	"a!b" /* ERROR invalid import path */
	"abc\xffdef" /* ERROR invalid import path */
)

// using "math" in this file doesn't affect its use in other files
const Pi0 = math.Pi
const Pi1 = m.Pi

type _ T // use "testing"

func _() func() interface{} {
	return func() interface{} {
		return Println // use "fmt"
	}
}
