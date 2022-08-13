// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo_test

import (
	"bytes"
	"io"
)

func getReader() io.Reader { return nil }

func do(b bytes.Reader) {}

func Example() {
	getReader()
	do()
	// Output:
}

func ExampleIgnored() {
}
