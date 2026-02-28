// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var s uint
var _ = string(1 /* ERROR shifted operand 1 .* must be integer */ << s)
