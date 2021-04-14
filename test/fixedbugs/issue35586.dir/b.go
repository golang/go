// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "a"

func F(addr string) (uint64, string) {
	return a.D(addr, 32)
}
