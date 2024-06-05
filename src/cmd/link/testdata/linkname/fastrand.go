// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linkname fastrand is allowed _for now_, as it has a
// linknamed definition, for legacy reason.
// NOTE: this may not be allowed in the future. Don't do this!

package main

import _ "unsafe"

//go:linkname fastrand runtime.fastrand
func fastrand() uint32

func main() {
	println(fastrand())
}
