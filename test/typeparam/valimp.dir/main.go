// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"a"
	"fmt"
)

func main() {
	var v1 a.Value[int]

	a.Set(&v1, 1)
	if got, want := a.Get(&v1), 1; got != want {
		panic(fmt.Sprintf("Get() == %d, want %d", got, want))
	}
	v1.Set(2)
	if got, want := v1.Get(), 2; got != want {
		panic(fmt.Sprintf("Get() == %d, want %d", got, want))
	}
	v1p := new(a.Value[int])
	a.Set(v1p, 3)
	if got, want := a.Get(v1p), 3; got != want {
		panic(fmt.Sprintf("Get() == %d, want %d", got, want))
	}

	v1p.Set(4)
	if got, want := v1p.Get(), 4; got != want {
		panic(fmt.Sprintf("Get() == %d, want %d", got, want))
	}

	var v2 a.Value[string]
	a.Set(&v2, "a")
	if got, want := a.Get(&v2), "a"; got != want {
		panic(fmt.Sprintf("Get() == %q, want %q", got, want))
	}

	v2.Set("b")
	if got, want := a.Get(&v2), "b"; got != want {
		panic(fmt.Sprintf("Get() == %q, want %q", got, want))
	}

	v2p := new(a.Value[string])
	a.Set(v2p, "c")
	if got, want := a.Get(v2p), "c"; got != want {
		panic(fmt.Sprintf("Get() == %d, want %d", got, want))
	}

	v2p.Set("d")
	if got, want := v2p.Get(), "d"; got != want {
		panic(fmt.Sprintf("Get() == %d, want %d", got, want))
	}
}
