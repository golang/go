// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package undeclared

func x() int {
	var z int
	z = y // want "(undeclared name|undefined): y"

	if z == m { // want "(undeclared name|undefined): m"
		z = 1
	}

	if z == 1 {
		z = 1
	} else if z == n+1 { // want "(undeclared name|undefined): n"
		z = 1
	}

	switch z {
	case 10:
		z = 1
	case a: // want "(undeclared name|undefined): a"
		z = 1
	}
	return z
}
