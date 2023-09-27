// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue20583

const (
	_ = 6e886451608 /* ERROR "malformed constant" */ /2
	_ = 6e886451608i /* ERROR "malformed constant" */ /2
	_ = 0 * 1e+1000000000 // ERROR "malformed constant"

	x = 1e100000000
	_ = x*x*x*x*x*x* /* ERROR "not representable" */ x
)
