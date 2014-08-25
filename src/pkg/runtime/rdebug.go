// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

func setMaxStack(in int) (out int) {
	out = int(maxstacksize)
	maxstacksize = uint(in)
	return out
}

func setGCPercent(in int32) (out int32) {
	mp := acquirem()
	mp.scalararg[0] = uint(int(in))
	onM(&setgcpercent_m)
	out = int32(int(mp.scalararg[0]))
	releasem(mp)
	return out
}

func setPanicOnFault(newb bool) (old bool) {
	new := uint8(0)
	if newb {
		new = 1
	}

	mp := acquirem()
	old = mp.curg.paniconfault == 1
	mp.curg.paniconfault = new
	releasem(mp)
	return old
}

func setMaxThreads(in int) (out int) {
	mp := acquirem()
	mp.scalararg[0] = uint(in)
	onM(&setmaxthreads_m)
	out = int(mp.scalararg[0])
	releasem(mp)
	return out
}
