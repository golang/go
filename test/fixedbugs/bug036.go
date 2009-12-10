// ! $G $D/$F.go >/dev/null
// # ignoring error messages...

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	s := float(0);
	s := float(0);  // BUG redeclaration
}
