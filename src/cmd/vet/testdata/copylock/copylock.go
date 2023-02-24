// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package copylock

import "sync"

func BadFunc() {
	var x *sync.Mutex
	p := x
	var y sync.Mutex
	p = &y
	*p = *x // ERROR "assignment copies lock value to \*p: sync.Mutex"
}
