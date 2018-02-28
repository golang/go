// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "iface_a"
import "iface_b"

func main() {
	if iface_a.F() != iface_b.F() {
		panic("empty interfaces not equal")
	}
	if iface_a.G() != iface_b.G() {
		panic("non-empty interfaces not equal")
	}
}
