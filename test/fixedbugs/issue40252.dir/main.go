// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

func main() {
	defer func() {
		if recover() == nil {
			panic("expected nil pointer dereference")
		}
	}()
	a.Call()
}
