// errorcheck

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	_ = string(-4 + 2i + 2) // ERROR "-4 \+ 2i"
}
