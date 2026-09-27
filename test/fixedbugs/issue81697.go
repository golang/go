// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var p *[3]int
	for i, _ := range *p {
		_ = i
	}
	for i, _ := range (*p) { // Note the parentheses! gofmt wants to remove them - don't let it!
		_ = i
	}
	var i int
	for i, (_) = range *p { // Note the parentheses! gofmt wants to remove them - don't let it!
		_ = i
	}
}
