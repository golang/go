// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iotest_test

import (
	"errors"
	"fmt"
	"testing/iotest"
)

func ExampleErrReader() {
	// A reader that always returns a custom error.
	r := iotest.ErrReader(errors.New("custom error"))
	n, err := r.Read(nil)
	fmt.Printf("n:   %d\nerr: %q\n", n, err)

	// Output:
	// n:   0
	// err: "custom error"
}
