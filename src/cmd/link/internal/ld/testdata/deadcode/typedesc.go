// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that a live variable doesn't bring its type
// descriptor live.

package main

type T [10]string

var t T

func main() {
	println(t[8])
}
