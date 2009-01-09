// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type T struct { a int; b string }

func (t *T) String() string {
	return fmt.sprint(t.a) + " " + t.b
}

func main() {
	t := &T{77, "Sunset Strip"};
	fmt.println(t)
}
