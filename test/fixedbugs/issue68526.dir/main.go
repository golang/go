// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.aliastypeparams

package main

import (
	"issue68526.dir/a"
)

func main() {
	unexported()
	// exported()
}

func unexported() {
	var want struct{ F int }

	if any(want) != any(a.B{}) || any(want) != any(a.F()) {
		panic("zero value of alias and concrete type not identical")
	}
}

// TODO(#68778): enable once type parameterized aliases are allowed in exportdata.

// func exported() {
// 	var (
// 		astr a.A[string]
// 		aint a.A[int]
// 	)

// 	if any(astr) != any(struct{ F string }{}) || any(aint) != any(struct{ F int }{}) {
// 		panic("zero value of alias and concrete type not identical")
// 	}

// 	if any(astr) == any(aint) {
// 		panic("zero value of struct{ F string } and struct{ F int } are not distinct")
// 	}

// 	if got := fmt.Sprintf("%T", astr); got != "struct { F string }" {
// 		panic(got)
// 	}
// }
