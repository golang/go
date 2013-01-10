// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Called from C. Returns the Go type *m.
func gc_m_ptr(ret *interface{}) {
	*ret = (*m)(nil)
}

// Called from C. Returns the Go type *itab.
func gc_itab_ptr(ret *interface{}) {
	*ret = (*itab)(nil)
}
