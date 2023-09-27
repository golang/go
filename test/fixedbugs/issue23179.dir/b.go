// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

func G(x int) int {
	return a.F(x, 1, false, a.Large{})
}
