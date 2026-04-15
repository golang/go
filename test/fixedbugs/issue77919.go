// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var sink any

func main() {
	i := 0
	output := make([]string, 8, i)
	sink = output
}
