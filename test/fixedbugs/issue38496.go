// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure bounds check elision isn't confused with nil check elision.

package main

func main() {
	defer func() {
		err := recover()
		if err == nil {
			panic("failed to check nil ptr")
		}
	}()
	var m [2]*int
	_ = *m[1] // need a nil check, but not a bounds check
}
