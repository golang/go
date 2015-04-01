// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd linux

package socktest

import "syscall"

// Accept4 wraps syscall.Accept4.
func (sw *Switch) Accept4(s, flags int) (ns int, sa syscall.Sockaddr, err error) {
	so := sw.sockso(s)
	if so == nil {
		return syscall.Accept4(s, flags)
	}
	sw.fmu.RLock()
	f, _ := sw.fltab[FilterAccept]
	sw.fmu.RUnlock()

	af, err := f.apply(so)
	if err != nil {
		return -1, nil, err
	}
	ns, sa, so.Err = syscall.Accept4(s, flags)
	if err = af.apply(so); err != nil {
		if so.Err == nil {
			syscall.Close(ns)
		}
		return -1, nil, err
	}

	if so.Err != nil {
		return -1, nil, so.Err
	}
	sw.smu.Lock()
	nso := sw.addLocked(ns, so.Cookie.Family(), so.Cookie.Type(), so.Cookie.Protocol())
	sw.stats.getLocked(nso.Cookie).Accepted++
	sw.smu.Unlock()
	return ns, sa, nil
}
