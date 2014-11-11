// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

func setMaxStack(in int) (out int) {
	out = int(maxstacksize)
	maxstacksize = uintptr(in)
	return out
}

func setPanicOnFault(new bool) (old bool) {
	mp := acquirem()
	old = mp.curg.paniconfault
	mp.curg.paniconfault = new
	releasem(mp)
	return old
}
