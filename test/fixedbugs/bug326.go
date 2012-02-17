// errorcheck

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() (_ int, err error) {
	return
}

func g() (x int, _ error) {
	return
}

func h() (_ int, _ error) {
	return
}

func i() (int, error) {
	return // ERROR "not enough arguments to return"
}

func f1() (_ int, err error) {
	return 1, nil
}

func g1() (x int, _ error) {
	return 1, nil
}

func h1() (_ int, _ error) {
	return 1, nil
}

func ii() (int, error) {
	return 1, nil
}
