// +build amd64
// errorcheck -0 -d=ssa/likelyadjust/debug=1

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that branches have some prediction properties.
package foo

func f(x, y, z int) int {
	a := 0
	for i := 0; i < x; i++ { // ERROR "Branch prediction rule stay in loop"
		for j := 0; j < y; j++ { // ERROR "Branch prediction rule stay in loop"
			a += j
		}
		for k := 0; k < z; k++ { // ERROR "Branch prediction rule stay in loop"
			a -= x + y + z
		}
	}
	return a
}

func g(x, y, z int) int {
	a := 0
	if y == 0 { // ERROR "Branch prediction rule default < call"
		y = g(y, z, x)
	} else {
		y++
	}
	if y == x { // ERROR "Branch prediction rule default < call"
		y = g(y, z, x)
	} else {
	}
	if y == 2 { // ERROR "Branch prediction rule default < call"
		z++
	} else {
		y = g(z, x, y)
	}
	if y+z == 3 { // ERROR "Branch prediction rule call < exit"
		println("ha ha")
	} else {
		panic("help help help")
	}
	if x != 0 { // ERROR "Branch prediction rule default < ret"
		for i := 0; i < x; i++ { // ERROR "Branch prediction rule stay in loop"
			if x == 4 { // ERROR "Branch prediction rule stay in loop"
				return a
			}
			for j := 0; j < y; j++ { // ERROR "Branch prediction rule stay in loop"
				for k := 0; k < z; k++ { // ERROR "Branch prediction rule stay in loop"
					a -= j * i
				}
				a += j
			}
		}
	}
	return a
}

func h(x, y, z int) int {
	a := 0
	for i := 0; i < x; i++ { // ERROR "Branch prediction rule stay in loop"
		for j := 0; j < y; j++ { // ERROR "Branch prediction rule stay in loop"
			a += j
			if i == j { // ERROR "Branch prediction rule stay in loop"
				break
			}
			a *= j
		}
		for k := 0; k < z; k++ { // ERROR "Branch prediction rule stay in loop"
			a -= k
			if i == k {
				continue
			}
			a *= k
		}
	}
	if a > 0 { // ERROR "Branch prediction rule default < call"
		a = g(x, y, z)
	} else {
		a = -a
	}
	return a
}
