// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func g() {
	var s string
	var i int
	_ = s /* ERROR invalid operation: s \+ i \(mismatched types string and int\) */ + i
}

func f(i int) int {
        i /* ERROR invalid operation: i \+= "1" \(mismatched types int and untyped string\) */ += "1"
        return i
}
