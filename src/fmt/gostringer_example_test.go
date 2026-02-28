// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	"fmt"
)

// Address has a City, State and a Country.
type Address struct {
	City    string
	State   string
	Country string
}

// Person has a Name, Age and Address.
type Person struct {
	Name string
	Age  uint
	Addr *Address
}

// GoString makes Person satisfy the GoStringer interface.
// The return value is valid Go code that can be used to reproduce the Person struct.
func (p Person) GoString() string {
	if p.Addr != nil {
		return fmt.Sprintf("Person{Name: %q, Age: %d, Addr: &Address{City: %q, State: %q, Country: %q}}", p.Name, int(p.Age), p.Addr.City, p.Addr.State, p.Addr.Country)
	}
	return fmt.Sprintf("Person{Name: %q, Age: %d}", p.Name, int(p.Age))
}

func ExampleGoStringer() {
	p1 := Person{
		Name: "Warren",
		Age:  31,
		Addr: &Address{
			City:    "Denver",
			State:   "CO",
			Country: "U.S.A.",
		},
	}
	// If GoString() wasn't implemented, the output of `fmt.Printf("%#v", p1)` would be similar to
	// Person{Name:"Warren", Age:0x1f, Addr:(*main.Address)(0x10448240)}
	fmt.Printf("%#v\n", p1)

	p2 := Person{
		Name: "Theia",
		Age:  4,
	}
	// If GoString() wasn't implemented, the output of `fmt.Printf("%#v", p2)` would be similar to
	// Person{Name:"Theia", Age:0x4, Addr:(*main.Address)(nil)}
	fmt.Printf("%#v\n", p2)

	// Output:
	// Person{Name: "Warren", Age: 31, Addr: &Address{City: "Denver", State: "CO", Country: "U.S.A."}}
	// Person{Name: "Theia", Age: 4}
}
