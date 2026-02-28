// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we can correctly compile method expressions
// where the method is implicitly declared.

package main

import "io"

func main() {
	err := io.EOF
	_ = err.Error
}
