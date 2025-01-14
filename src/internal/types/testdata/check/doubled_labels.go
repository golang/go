// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
outer:
inner:
	for {
		continue inner
		break inner
	}
	goto outer
}

func _() {
outer:
inner:
	for {
		continue inner
		continue outer /* ERROR "invalid continue label outer" */
		break outer    /* ERROR "invalid break label outer" */
	}
	goto outer
}
