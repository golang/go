// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	var ch chan bool
	select {
	default:
	case <-ch { // don't crash here
	}           // ERROR "expecting :"
}
