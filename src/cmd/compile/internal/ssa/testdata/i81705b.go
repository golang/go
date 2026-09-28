// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func typeSwitch(x any) {
	switch x.(type) {
	case interface{ M() }:
		println("M")
	default:
		println("default")
	}
}

func main() {
	typeSwitch(1)
}
