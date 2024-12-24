// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that invalid identifiers reported by the parser
// don't lead to additional errors during typechecking.

package p

import "fmt"

var (
	☹x /* ERROR "invalid character" */ int
	_ = ☹x // ERROR "invalid character"
	_ = fmt.☹x // ERROR "invalid character"
	_ = ☹fmt /* ERROR "invalid character" */ .Println
	_ = _世界 // ERROR "undefined: _世界"
	_ = ☹_世界 // ERROR "invalid character"
)

func ☹m /* ERROR "invalid character" */ () {}

type T struct{}
func (T) ☹m /* ERROR "invalid character" */ () {}

func _() {
	var x T
	x.☹m /* ERROR "invalid character" */ ()
}
