// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 11737 - invalid == not being caught until generated switch code was compiled

package p

func f()

func s(x interface{}) {
	switch x {
	case f: // ERROR "invalid case f \(type func\(\)\) in switch \(incomparable type\)"
	}
}
