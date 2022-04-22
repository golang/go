// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p_test

import (
	"fmt"
	"time"
)

type C1 interface {
	string | int
}

type C2 interface {
	M(time.Time)
}

type G[T C1] int

func g[T C2](x T) {}

type Tm int

func (Tm) M(time.Time) {}

type Foo int

func Example() {
	fmt.Println("hello")
}

func ExampleGeneric() {
	var x G[string]
	g(Tm(3))
	fmt.Println(x)
}
