// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo_test

import (
	"fmt"
	"time"

	"golang.org/x/time/rate"
)

func Example() {
	fmt.Println("Hello, world!")
	// Output: Hello, world!
}

func ExampleLimiter() {
	// Uses fmt, time and rate.
	l := rate.NewLimiter(rate.Every(time.Second), 1)
	fmt.Println(l)
}
