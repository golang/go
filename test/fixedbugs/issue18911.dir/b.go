// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"
import "strings"

func main() {
	defer func() {
		p, ok := recover().(error)
		if ok && strings.Contains(p.Error(), "different packages") {
			return
		}
		panic(p)
	}()

	// expected to fail and report two identical looking (but different) types
	_ = a.X.(struct{ x int })
}
