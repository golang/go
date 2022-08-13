// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var _ = (*interface /* ERROR interface contains type constraints */ {int})(nil)

// abbreviated test case from issue

type TypeSet interface{ int | string }

func _() {
	f((*TypeSet /* ERROR interface contains type constraints */)(nil))
}

func f(any) {}