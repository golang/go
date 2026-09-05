// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unique_test

import (
	"fmt"

	"unique"
)

func ExampleMake() {
	type user struct {
		Name string
		Age  int
	}

	// Handles for equal values are equal, even when the values were
	// constructed independently.
	alice1 := user{Name: "Alice", Age: 30}
	alice2 := user{Name: "Alice", Age: 30}
	bob := user{Name: "Bob", Age: 30}

	aliceHandle1 := unique.Make(alice1)
	aliceHandle2 := unique.Make(alice2)
	bobHandle := unique.Make(bob)

	// The original values can be discarded, while the interned value remains
	// available through the handle.
	alice1 = user{}
	alice2 = user{}
	bob = user{}

	fmt.Println(aliceHandle1 == aliceHandle2)
	fmt.Println(aliceHandle1 == bobHandle)
	fmt.Println(aliceHandle1.Value())

	// Output:
	// true
	// false
	// {Alice 30}
}
