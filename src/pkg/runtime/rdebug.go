// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

func setMaxStack(in int) (out int) {
	out = int(maxstacksize)
	maxstacksize = uintptr(in)
	return out
}

func setGCPercent(in int32) (out int32) {
	mp := acquirem()
	mp.scalararg[0] = uintptr(int(in))
	onM(setgcpercent_m)
	out = int32(int(mp.scalararg[0]))
	releasem(mp)
	return out
}

func setPanicOnFault(new bool) (old bool) {
	mp := acquirem()
	old = mp.curg.paniconfault
	mp.curg.paniconfault = new
	releasem(mp)
	return old
}

func setMaxThreads(in int) (out int) {
	mp := acquirem()
	mp.scalararg[0] = uintptr(in)
	onM(setmaxthreads_m)
	out = int(mp.scalararg[0])
	releasem(mp)
	return out
}
