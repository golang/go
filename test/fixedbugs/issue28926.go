// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var e interface{}
	switch e := e.(type) {
	case G: // ERROR "undefined: G"
		e.M() // [this error should be ignored because the case failed its typecheck]
	case E: // ERROR "undefined: E"
		e.D() // [this error should be ignored because the case failed its typecheck]
	}
}
