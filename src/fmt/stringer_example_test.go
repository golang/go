// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	"fmt"
)

// Animal has a Name and an Age to represent an animal.
type Animal struct {
	Name string
	Age  uint
}

// String makes Animal satisfy the Stringer interface.
func (a Animal) String() string {
	return fmt.Sprintf("%v (%d)", a.Name, a.Age)
}

func ExampleStringer() {
	a := Animal{
		Name: "Gopher",
		Age:  2,
	}
	fmt.Println(a)
	// Output: Gopher (2)
}
