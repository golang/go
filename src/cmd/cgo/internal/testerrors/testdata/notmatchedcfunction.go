// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
// TODO(#56378): change back to "#cgo noescape noMatchedCFunction: no matched C function" in Go 1.23
// ERROR MESSAGE: #cgo noescape disabled until Go 1.23
#cgo noescape noMatchedCFunction
*/
import "C"

func main() {
}
