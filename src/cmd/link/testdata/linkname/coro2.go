// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linkname corostart is not allowed, as it doesn't have
// a linknamed definition.

package main

import _ "unsafe"

//go:linkname corostart runtime.corostart
func corostart()

func main() {
	corostart()
}
