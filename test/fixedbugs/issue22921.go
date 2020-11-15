// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "bytes"

type _ struct{ bytes.nonexist } // ERROR "unexported"

type _ interface{ bytes.nonexist } // ERROR "unexported"

func main() {
	var _ bytes.Buffer
	var _ bytes.buffer // ERROR "unexported"
}
