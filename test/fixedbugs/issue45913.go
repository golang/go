// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

func f(s1, s2 string) { fmt.Printf("%s %s", s1, s2) }

func main() {
	f([2]string{"a", "b"}...) // ERROR "invalid use of .*[.][.][.]|cannot use [.][.][.] in call to non-variadic"
}
