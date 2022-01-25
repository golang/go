// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errors_test

import (
	"errors"
	"fmt"
)

func ExampleMark() {
	err := errors.Mark(nil, nil)
	fmt.Println(err)
	fmt.Println("-----")
	err = errors.Mark(fmt.Errorf("whoops"), nil)
	fmt.Println(err)
	fmt.Println("-----")
	err = errors.Mark(fmt.Errorf("whoops"), fmt.Errorf("foo"))
	fmt.Println(err)
	fmt.Println("-----")
	err = errors.Mark(fmt.Errorf("whoops"), fmt.Errorf("foo"), fmt.Errorf("bar"))
	fmt.Println(err)
	fmt.Println("-----")

	// Output:
	// <nil>
	// -----
	// whoops
	// -----
	// whoops
	// -----
	// whoops
	// -----
}

func ExampleMark_format() {
	err := errors.Mark(nil)
	fmt.Printf("v: %v\n", err)
	fmt.Printf("+v: %+v\n", err)
	fmt.Println("-----")

	err = errors.Mark(fmt.Errorf("whoops"), nil)
	fmt.Printf("v: %v\n", err)
	fmt.Printf("+v: %+v\n", err)
	fmt.Println("-----")

	err = errors.Mark(fmt.Errorf("whoops"), fmt.Errorf("foo"))
	fmt.Printf("v: %v\n", err)
	fmt.Printf("+v: %+v\n", err)
	fmt.Println("-----")

	err = errors.Mark(fmt.Errorf("whoops"), fmt.Errorf("foo"), fmt.Errorf("bar"))
	fmt.Printf("v: %v\n", err)
	fmt.Printf("+v: %+v\n", err)
	fmt.Println("-----")

	// Output:
	// v: <nil>
	// +v: <nil>
	// -----
	// v: whoops
	// +v: whoops
	// -----
	// v: whoops
	// +v: Marked errors occurred:
	// |	whoops
	// M	foo
	// -----
	// v: whoops
	// +v: Marked errors occurred:
	// |	whoops
	// M	foo
	// M	bar
	// -----
}

func ExampleMark_is() {
	var mark = errors.New("mark")
	err := errors.Mark(nil, mark)
	fmt.Printf("%v\n", errors.Is(err, nil))
	fmt.Printf("%v\n", errors.Is(err, mark))
	fmt.Println("-----")

	err = errors.Mark(fmt.Errorf("whoops"), nil, mark)
	fmt.Printf("%v\n", errors.Is(err, nil))
	fmt.Printf("%v\n", errors.Is(err, mark))
	fmt.Println("-----")

	err = errors.Mark(fmt.Errorf("whoops"), fmt.Errorf("foo"), mark)
	fmt.Printf("%v\n", errors.Is(err, nil))
	fmt.Printf("%v\n", errors.Is(err, mark))
	fmt.Println("-----")

	err = errors.Mark(fmt.Errorf("whoops"), fmt.Errorf("foo"), fmt.Errorf("bar"), mark)
	fmt.Printf("%v\n", errors.Is(err, nil))
	fmt.Printf("%v\n", errors.Is(err, mark))
	fmt.Println("-----")

	// Output:
	// true
	// false
	// -----
	// false
	// true
	// -----
	// false
	// true
	// -----
	// false
	// true
	// -----

}
