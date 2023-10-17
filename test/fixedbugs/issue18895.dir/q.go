// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package q

import "./p"

func x() { // ERROR "can inline x"
	p.F() // ERROR "inlining call to .*\.F" "inlining call to .*\.m"
}
