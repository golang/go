// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
	"fmt"
)

func main() {
	const want = 2
	if got := a.Min[int](2, 3); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	if got := a.Min(2, 3); got != want {
		panic(fmt.Sprintf("want %d, got %d", want, got))
	}

	if got := a.Min[float64](3.5, 2.0); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	if got := a.Min(3.5, 2.0); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	const want2 = "ay"
	if got := a.Min[string]("bb", "ay"); got != want2 { // ERROR "string does not satisfy"
		panic(fmt.Sprintf("got %d, want %d", got, want2))
	}

	if got := a.Min("bb", "ay"); got != want2 { // ERROR "string does not satisfy"
		panic(fmt.Sprintf("got %d, want %d", got, want2))
	}
}
