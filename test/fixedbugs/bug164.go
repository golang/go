// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Multi-line string literal now allowed.

const s = `
Hello, World!
`

func main() {
	print(s)
}
