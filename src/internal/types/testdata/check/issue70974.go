// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
outer:
	for {
		break outer
	}

	for {
		break outer /* ERROR "invalid break label outer" */
	}
}

func _() {
outer:
	for {
		continue outer
	}

	for {
		continue outer /* ERROR "invalid continue label outer" */
	}
}
