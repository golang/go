// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for issue 23434: Better synchronization of
// parser after missing type. There should be exactly
// one error each time, with now follow errors.

package p

type T /* ERROR unexpected newline */

type Map map[int] /* ERROR unexpected newline */

// Examples from #23434:

func g() {
	m := make(map[string] /* ERROR unexpected ! */ !)
	for {
		x := 1
		print(x)
	}
}

func f() {
	m := make(map[string] /* ERROR unexpected \) */ )
	for {
		x := 1
		print(x)
	}
}
