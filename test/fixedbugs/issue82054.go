// build

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func F() func() {
	type T struct{ x int }
	t := &T{}
	return func() {
		go func() { t.x = 1 }()
	}
}
