// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the try builtin in various forms.

package main

import "fmt"

type Test struct {
	name string
}

func test(name string) *Test {
	return &Test{name: name}
}

func (t *Test) expect(expected ...interface{}) func(...interface{}) {
	return func(actual ...interface{}) {
		if len(actual) != len(expected) {
			fmt.Printf("%v: expected %v elements, got %v", t.name, len(expected), len(actual))
			return
		}

		for i := 0; i < len(actual); i++ {
			if actual[i] != expected[i] {
				fmt.Printf("%v: expected %v, got %v for element %v\n", t.name, expected[i], actual[i], i)
			}
		}
	}
}

var Err = fmt.Errorf("error")
var Err2 = fmt.Errorf("error 2")

func e(err error) error {
	return err
}

func ie(err error) (int, error) {
	return 1, err
}

func be(err error) (bool, error) {
	return true, err
}

func iie(err error) (int, int, error) {
	return 1, 2, err
}

func main() {
	test("namedReturnValues1").expect(1, nil)(func() (n int, err error) {
		n = try(ie(nil))
		return
	}())

	test("namedReturnValues1_err").expect(0, Err)(func() (n int, err error) {
		n = try(ie(Err))
		err = nil
		return
	}())

	test("namedReturnValues1_err_preset").expect(-1, Err)(func() (n int, err error) {
		n = -1
		n = try(ie(Err))
		return
	}())

	test("namedReturnValues2").expect(1, 2, nil)(func() (n, m int, err error) {
		n, m = try(iie(nil))
		return
	}())

	test("namedReturnValues2_err").expect(0, 0, Err)(func() (n, m int, err error) {
		n, m = try(iie(Err))
		err = nil
		return
	}())

	test("unnamedReturnValues1").expect(1, nil)(func() (int, error) {
		n := try(ie(nil))
		return n, nil
	}())

	test("unnamedReturnValues1_err").expect(0, Err)(func() (int, error) {
		n := try(ie(Err))
		return n, nil
	}())

	test("unnamedReturnValues2").expect(1, 2, nil)(func() (int, int, error) {
		n, m := try(iie(nil))
		return n, m, nil
	}())

	test("unnamedReturnValues2_err").expect(0, 0, Err)(func() (int, int, error) {
		n, m := try(iie(Err))
		return n, m, nil
	}())

	test("assignOp").expect(1, nil)(func() (n int, err error) {
		n += try(ie(nil))
		return
	}())

	test("binaryOp").expect(2, nil)(func() (n int, err error) {
		return 1 + try(ie(nil)), nil
	}())

	test("chain").expect(2, 4, nil)(func() (int, int, error) {
		n, m := try(func(n, m int) (int, int, error) {
			return n + 1, m + 2, nil
		}(try(iie(nil))))
		return n, m, nil
	}())

	test("chain_err").expect(0, 0, Err)(func() (int, int, error) {
		n, m := try(func(n, m int) (int, int, error) {
			return n + 1, m + 2, Err
		}(try(iie(nil))))
		return n, m, nil
	}())

	test("defer").expect(0, Err2)(func() (n int, err error) {
		defer func() {
			if err != nil {
				err = Err2
			}
		}()
		return try(ie(Err)), nil
	}())

	test("binaryTryOps").expect(2, nil)(func() (int, error) {
		return try(ie(nil)) + try(ie(nil)), nil
	}())

	test("errorOnly").expect(2, nil)(func() (int, error) {
		try(e(nil))
		return 2, nil
	}())

	test("errorOnly_err").expect(0, Err)(func() (int, error) {
		try(e(Err))
		return 2, nil
	}())

	test("errorOnly_err_var").expect(0, Err)(func() (int, error) {
		try(Err)
		return 2, nil
	}())
}
