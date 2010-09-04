// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func assert(cond bool, msg string) {
	if !cond {
		print("assertion fail: ", msg, "\n")
		panic(1)
	}
}

func main() {
	i5 := 5
	i7 := 7
	hello := "hello"

	switch true {
	case i5 < 5: assert(false, "<")
	case i5 == 5: assert(true, "!")
	case i5 > 5: assert(false, ">")
	}

	switch {
	case i5 < 5: assert(false, "<")
	case i5 == 5: assert(true, "!")
	case i5 > 5: assert(false, ">")
	}

	switch x := 5; true {
	case i5 < x: assert(false, "<")
	case i5 == x: assert(true, "!")
	case i5 > x: assert(false, ">")
	}

	switch x := 5; true {
	case i5 < x: assert(false, "<")
	case i5 == x: assert(true, "!")
	case i5 > x: assert(false, ">")
	}

	switch i5 {
	case 0: assert(false, "0")
	case 1: assert(false, "1")
	case 2: assert(false, "2")
	case 3: assert(false, "3")
	case 4: assert(false, "4")
	case 5: assert(true, "5")
	case 6: assert(false, "6")
	case 7: assert(false, "7")
	case 8: assert(false, "8")
	case 9: assert(false, "9")
	default: assert(false, "default")
	}

	switch i5 {
	case 0,1,2,3,4: assert(false, "4")
	case 5: assert(true, "5")
	case 6,7,8,9: assert(false, "9")
	default: assert(false, "default")
	}

	switch i5 {
	case 0:
	case 1:
	case 2:
	case 3:
	case 4: assert(false, "4")
	case 5: assert(true, "5")
	case 6:
	case 7:
	case 8:
	case 9:
	default: assert(i5 == 5, "good")
	}

	switch i5 {
	case 0: dummy := 0; _ = dummy; fallthrough
	case 1: dummy := 0; _ = dummy; fallthrough
	case 2: dummy := 0; _ = dummy; fallthrough
	case 3: dummy := 0; _ = dummy; fallthrough
	case 4: dummy := 0; _ = dummy; assert(false, "4")
	case 5: dummy := 0; _ = dummy; fallthrough
	case 6: dummy := 0; _ = dummy; fallthrough
	case 7: dummy := 0; _ = dummy; fallthrough
	case 8: dummy := 0; _ = dummy; fallthrough
	case 9: dummy := 0; _ = dummy; fallthrough
	default: dummy := 0; _ = dummy; assert(i5 == 5, "good")
	}

	fired := false
	switch i5 {
	case 0: dummy := 0; _ = dummy; fallthrough;  // tests scoping of cases
	case 1: dummy := 0; _ = dummy; fallthrough
	case 2: dummy := 0; _ = dummy; fallthrough
	case 3: dummy := 0; _ = dummy; fallthrough
	case 4: dummy := 0; _ = dummy; assert(false, "4")
	case 5: dummy := 0; _ = dummy; fallthrough
	case 6: dummy := 0; _ = dummy; fallthrough
	case 7: dummy := 0; _ = dummy; fallthrough
	case 8: dummy := 0; _ = dummy; fallthrough
	case 9: dummy := 0; _ = dummy; fallthrough
	default: dummy := 0; _ = dummy; fired = !fired; assert(i5 == 5, "good")
	}
	assert(fired, "fired")

	count := 0
	switch i5 {
	case 0: count = count + 1; fallthrough
	case 1: count = count + 1; fallthrough
	case 2: count = count + 1; fallthrough
	case 3: count = count + 1; fallthrough
	case 4: count = count + 1; assert(false, "4")
	case 5: count = count + 1; fallthrough
	case 6: count = count + 1; fallthrough
	case 7: count = count + 1; fallthrough
	case 8: count = count + 1; fallthrough
	case 9: count = count + 1; fallthrough
	default: assert(i5 == count, "good")
	}
	assert(fired, "fired")

	switch hello {
	case "wowie": assert(false, "wowie")
	case "hello": assert(true, "hello")
	case "jumpn": assert(false, "jumpn")
	default: assert(false, "default")
	}

	fired = false
	switch i := i5 + 2; i {
	case i7: fired = true
	default: assert(false, "fail")
	}
	assert(fired, "var")
}
