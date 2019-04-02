// errorcheck -0 -N -m -l -newescape=false

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The escape analyzer needs to run till its root set settles
// (this is not that often, it turns out).
// This test is likely to become stale because the leak depends
// on a spurious-escape bug -- return an interface as a named
// output parameter appears to cause the called closure to escape,
// where returning it as a regular type does not.

package main

import (
	"fmt"
)

type closure func(i, j int) ent

type ent int

func (e ent) String() string {
	return fmt.Sprintf("%d", int(e)) // ERROR "ent.String ... argument does not escape$" "int\(e\) escapes to heap$"
}

//go:noinline
func foo(ops closure, j int) (err fmt.Stringer) { // ERROR "leaking param: ops$" "leaking param: ops to result err level=0$"
	enqueue := func(i int) fmt.Stringer { // ERROR "func literal escapes to heap$"
		return ops(i, j) // ERROR "ops\(i, j\) escapes to heap$"
	}
	err = enqueue(4)
	if err != nil {
		return err
	}
	return // return result of enqueue, a fmt.Stringer
}

func main() {
	// 3 identical functions, to get different escape behavior.
	f := func(i, j int) ent { // ERROR "func literal escapes to heap$"
		return ent(i + j)
	}
	i := foo(f, 3).(ent)
	fmt.Printf("foo(f,3)=%d\n", int(i)) // ERROR "int\(i\) escapes to heap$" "main ... argument does not escape$"
}
