// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 12536: compiler crashes while checking keys in a map literal for equality

package p

func main() {
	m1 := map[interface{}]interface{}{
		nil:  0,
		true: 1,
	}
	m2 := map[interface{}]interface{}{
		true: 1,
		nil:  0,
	}
	println(len(m1))
	println(len(m2))
}
