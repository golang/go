// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test simple function literals.

package main

func
main() {
	x := func(a int)int {
		x := func(a int)int {
			x := func(a int)int {
				return a+5;
			};
			return x(a)+7;
		};
		return x(a)+11;
	};
	if x(3) != 3+5+7+11 { panic(x(3)); }
}
