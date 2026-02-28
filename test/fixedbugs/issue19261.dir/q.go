// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package q

import "./p"

func H() {
	p.F() // ERROR "inlining call to p.F"
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
}
