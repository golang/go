// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// check that initialization loop is diagnosed
// and that closure cannot be used to hide it.
// error message is not standard format, so no errchk above.

package main

type F func()

func f() {
	if true {
		_ = func() { _ = m }
	}
}

var m = map[string]F{"f": f} // ERROR "initialization loop|depends upon itself|initialization cycle"
