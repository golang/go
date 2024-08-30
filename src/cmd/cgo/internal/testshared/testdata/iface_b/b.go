// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iface_b

import "testshared/iface_i"

//go:noinline
func F() interface{} {
	return (*iface_i.T)(nil)
}

//go:noinline
func G() iface_i.I {
	return (*iface_i.T)(nil)
}
