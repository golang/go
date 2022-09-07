// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

func F() any {
	return struct{ int }{0}
}

func main() {
	_, ok1 := F().(struct{ int })
	_, ok2 := a.F().(struct{ int })
	if !ok1 || ok2 {
		panic(0)
	}
}
