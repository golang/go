// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

func main() {
	a.One(nil)
	Two(nil)
}

func Two(L any) {
	func() {
		defer a.F(L)
	}()
}
