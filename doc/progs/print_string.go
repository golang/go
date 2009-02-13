// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type testType struct { a int; b string }

func (t *testType) String() string {
	return fmt.Sprint(t.a) + " " + t.b
}

func main() {
	t := &testType(77, "Sunset Strip");
	fmt.Println(t)
}
