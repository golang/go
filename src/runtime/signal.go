// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

func sigenable_m() {
	_g_ := getg()
	sigenable(uint32(_g_.m.scalararg[0]))
}

func sigdisable_m() {
	_g_ := getg()
	sigdisable(uint32(_g_.m.scalararg[0]))
}
