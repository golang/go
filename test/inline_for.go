// errorcheck -0 -m=2

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test, using compiler diagnostic flags, that inlining is working.
// Compiles but does not run.

package foo

import "runtime"

func func_with() int { // ERROR "can inline func_with .*"
	return 10
}

func func_with_cost_88() { // ERROR "can inline only into small FORs .*"
	x := 200
	for i := 0; i < x; i++ { // ERROR "add FOR to stack \[\{39\}\]"
		if i%2 == 0 {
			runtime.GC()
		} else {
			i += 2
			x += 1
		}
	}
}

func func_with_fors() { // ERROR "can inline .*"
	for { // ERROR "add FOR to stack \[\{22\}\]"
		for { // ERROR "add FOR to stack \[\{22\} \{16\}\]"
			func_with_cost_88() // ERROR "inlining call to func_with_cost_88" "add FOR to stack \[\{22\} \{16\} \{39\}\]"
		}
		for { // ERROR "add FOR to stack"
			func_with() // ERROR "inlining call to func_with"
		}
	}
}
