// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f1[T1, T2 any](T1, T2, struct{a T1; b T2}) {}
func _() {
	f1(42, string("foo"), struct /* ERROR does not match inferred type struct\{a int; b string\} */ {a, b int}{})
}

// simplified test case from issue
func f2[T any](_ []T, _ func(T)) {}
func _() {
	f2([]string{}, func /* ERROR does not match inferred type func\(string\) */ (f []byte) {})
}
