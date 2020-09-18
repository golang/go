// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lostcancel

import "context"

func _() {
	var _, cancel = context.WithCancel(context.Background()) // ERROR "the cancel function is not used on all paths \(possible context leak\)"
	if false {
		_ = cancel
	}
} // ERROR "this return statement may be reached without using the cancel var defined on line 10"
