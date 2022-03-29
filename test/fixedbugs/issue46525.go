// errorcheck -lang=go1.16

// Copyright 2021 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package p

import "unsafe"

func main() {
	_ = unsafe.Add(unsafe.Pointer(nil), 0) // ERROR "unsafe.Add requires go1.17 or later"
	_ = unsafe.Slice(new(byte), 1)         // ERROR "unsafe.Slice requires go1.17 or later"
}
