// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// No `"fmt" imported and not used` error below.
// The switch cases must be typechecked even
// though the switch expression is invalid.

import "fmt"

func _() {
	x := 1
	for e := range x.m /* ERROR "x.m undefined (type int has no field or method m)" */ () {
		switch e.(type) {
		case int:
			fmt.Println()
		}
	}

	switch t := x /* ERROR "not an interface" */ .(type) {
	case int, string:
		_ = t
	}
}
