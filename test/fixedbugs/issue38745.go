// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type t struct{ x int }

func f1() {
	t{}.M()     // ERROR "t{}.M undefined \(type t has no field or method M\)|undefined field or method .*M"
	t{x: 1}.M() // ERROR "t{...}.M undefined \(type t has no field or method M\)|undefined field or method .*M|no field or method M"
}

func f2() (*t, error) {
	return t{}.M() // ERROR "t{}.M undefined \(type t has no field or method M\)|undefined field or method .*M|not enough arguments"
}
