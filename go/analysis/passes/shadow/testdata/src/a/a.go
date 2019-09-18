// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the shadowed variable checker.
// Some of these errors are caught by the compiler (shadowed return parameters for example)
// but are nonetheless useful tests.

package a

import "os"

func ShadowRead(f *os.File, buf []byte) (err error) {
	var x int
	if f != nil {
		err := 3 // OK - different type.
		_ = err
	}
	if f != nil {
		_, err := f.Read(buf) // want "declaration of .err. shadows declaration at line 13"
		if err != nil {
			return err
		}
		i := 3 // OK
		_ = i
	}
	if f != nil {
		x := one()               // want "declaration of .x. shadows declaration at line 14"
		var _, err = f.Read(buf) // want "declaration of .err. shadows declaration at line 13"
		if x == 1 && err != nil {
			return err
		}
	}
	for i := 0; i < 10; i++ {
		i := i // OK: obviously intentional idiomatic redeclaration
		go func() {
			println(i)
		}()
	}
	var shadowTemp interface{}
	switch shadowTemp := shadowTemp.(type) { // OK: obviously intentional idiomatic redeclaration
	case int:
		println("OK")
		_ = shadowTemp
	}
	if shadowTemp := shadowTemp; true { // OK: obviously intentional idiomatic redeclaration
		var f *os.File // OK because f is not mentioned later in the function.
		// The declaration of x is a shadow because x is mentioned below.
		var x int // want "declaration of .x. shadows declaration at line 14"
		_, _, _ = x, f, shadowTemp
	}
	// Use a couple of variables to trigger shadowing errors.
	_, _ = err, x
	return
}

func one() int {
	return 1
}

// Must not complain with an internal error for the
// implicitly declared type switch variable v.
func issue26725(x interface{}) int {
	switch v := x.(type) {
	case int, int32:
		if v, ok := x.(int); ok {
			return v
		}
	case int64:
		return int(v)
	}
	return 0
}

// Verify that implicitly declared variables from
// type switches are considered in shadowing analysis.
func shadowTypeSwitch(a interface{}) {
	switch t := a.(type) {
	case int:
		{
			t := 0 // want "declaration of .t. shadows declaration at line 78"
			_ = t
		}
		_ = t
	case uint:
		{
			t := uint(0) // OK because t is not mentioned later in this function
			_ = t
		}
	}
}

func shadowBlock() {
	var a int
	{
		var a = 3 // want "declaration of .a. shadows declaration at line 94"
		_ = a
	}
	_ = a
}
