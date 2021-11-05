// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[P int](x P) int {
	return x // ERROR cannot use x .* as int value in return statement
}

func _[P int]() int {
	return P /* ERROR cannot use P\(1\) .* as int value in return statement */ (1)
}

func _[P int](x int) P {
        return x // ERROR cannot use x .* as P value in return statement
}

func _[P, Q any](x P) Q {
        return x // ERROR cannot use x .* as Q value in return statement
}

// test case from issue
func F[G interface{ uint }]() int {
	f := func(uint) int { return 0 }
	return f(G /* ERROR cannot use G\(1\) .* as uint value in argument to f */ (1))
}
