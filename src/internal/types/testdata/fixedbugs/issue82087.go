// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// We should get at most one error per line.

var s int
var _ int8 = ( /* ERROR "overflows" */ 1000)
var _ int8 = (1000 /* ERROR "overflows int8" */ ) << s
var _ int8 = ((1000 /* ERROR "overflows int8" */ )) << s
var _ int8 = (((1000 /* ERROR "overflows int8" */ ))) << s
