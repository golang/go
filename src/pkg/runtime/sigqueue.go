// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements runtime support for signal handling.

package runtime

func signal_recv() (m uint32) {
	for {
		mp := acquirem()
		onM(signal_recv_m)
		ok := mp.scalararg[0] != 0
		m = uint32(mp.scalararg[1])
		releasem(mp)
		if ok {
			return
		}
		notetsleepg(&signote, -1)
		noteclear(&signote)
	}
}

func signal_enable(s uint32) {
	mp := acquirem()
	mp.scalararg[0] = uintptr(s)
	onM(signal_enable_m)
	releasem(mp)
}

func signal_disable(s uint32) {
	mp := acquirem()
	mp.scalararg[0] = uintptr(s)
	onM(signal_disable_m)
	releasem(mp)
}

func signal_recv_m()
func signal_enable_m()
func signal_disable_m()
