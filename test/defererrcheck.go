// errorcheck -0 -l -d=defer

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// check that open-coded defers are used in expected situations

package main

import "fmt"

var glob = 3

func f1() {

	for i := 0; i < 10; i++ {
		fmt.Println("loop")
	}
	defer func() { // ERROR "open-coded defer"
		fmt.Println("defer")
	}()
}

func f2() {
	for {
		defer func() { // ERROR "heap-allocated defer"
			fmt.Println("defer1")
		}()
		if glob > 2 {
			break
		}
	}
	defer func() { // ERROR "stack-allocated defer"
		fmt.Println("defer2")
	}()
}

func f3() {
	defer func() { // ERROR "stack-allocated defer"
		fmt.Println("defer2")
	}()
	for {
		defer func() { // ERROR "heap-allocated defer"
			fmt.Println("defer1")
		}()
		if glob > 2 {
			break
		}
	}
}

func f4() {
	defer func() { // ERROR "open-coded defer"
		fmt.Println("defer")
	}()
label:
	fmt.Println("goto loop")
	if glob > 2 {
		goto label
	}
}

func f5() {
label:
	fmt.Println("goto loop")
	defer func() { // ERROR "heap-allocated defer"
		fmt.Println("defer")
	}()
	if glob > 2 {
		goto label
	}
}

func f6() {
label:
	fmt.Println("goto loop")
	if glob > 2 {
		goto label
	}
	// The current analysis doesn't end a backward goto loop, so this defer is
	// considered to be inside a loop
	defer func() { // ERROR "heap-allocated defer"
		fmt.Println("defer")
	}()
}

// Test for function with too many exits, which will disable open-coded defer
// even though the number of defer statements is not greater than 8.
func f7() {
	defer println(1) // ERROR "open-coded defer"
	defer println(1) // ERROR "open-coded defer"
	defer println(1) // ERROR "open-coded defer"
	defer println(1) // ERROR "open-coded defer"

	switch glob {
	case 1:
		return
	case 2:
		return
	case 3:
		return
	}
}

func f8() {
	defer println(1) // ERROR "stack-allocated defer"
	defer println(1) // ERROR "stack-allocated defer"
	defer println(1) // ERROR "stack-allocated defer"
	defer println(1) // ERROR "stack-allocated defer"

	switch glob {
	case 1:
		return
	case 2:
		return
	case 3:
		return
	case 4:
		return
	}
}

func f9() {
	defer println(1) // ERROR "open-coded defer"
	defer println(1) // ERROR "open-coded defer"
	defer println(1) // ERROR "open-coded defer"
	defer println(1) // ERROR "open-coded defer"

	switch glob {
	case 1:
		return
	case 2:
		return
	case 3:
		return
	case 4:
		panic("")
	}
}
