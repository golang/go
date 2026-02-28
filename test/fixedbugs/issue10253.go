// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 10253: cmd/7g: bad codegen, probably regopt related

package main

func main() {
	if !eq() {
		panic("wrong value")
	}
}

var text = "abc"
var s = &str{text}

func eq() bool {
	return text[0] == s.text[0]
}

type str struct {
	text string
}
