// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo_test

import (
	"fmt"

	"example.com/error"
)

func Print(s string) {
	fmt.Println(s)
}

func Example() {
	Print(error.Hello)
}
