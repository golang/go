// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"a"
	"fmt"
)

func main() {
	if got, want := a.OrderedAbsDifference(1.0, -2.0), 3.0; got != want {
		panic(fmt.Sprintf("got = %v, want = %v", got, want))
	}
	if got, want := a.OrderedAbsDifference(-1.0, 2.0), 3.0; got != want {
		panic(fmt.Sprintf("got = %v, want = %v", got, want))
	}
	if got, want := a.OrderedAbsDifference(-20, 15), 35; got != want {
		panic(fmt.Sprintf("got = %v, want = %v", got, want))
	}

	if got, want := a.ComplexAbsDifference(5.0+2.0i, 2.0-2.0i), 5+0i; got != want {
		panic(fmt.Sprintf("got = %v, want = %v", got, want))
	}
	if got, want := a.ComplexAbsDifference(2.0-2.0i, 5.0+2.0i), 5+0i; got != want {
		panic(fmt.Sprintf("got = %v, want = %v", got, want))
	}
}
