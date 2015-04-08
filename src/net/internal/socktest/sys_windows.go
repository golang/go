// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socktest

import "syscall"

// Socket wraps syscall.Socket.
func (sw *Switch) Socket(family, sotype, proto int) (s syscall.Handle, err error) {
	so := &Status{Cookie: cookie(family, sotype, proto)}
	sw.fmu.RLock()
	f, _ := sw.fltab[FilterSocket]
	sw.fmu.RUnlock()

	af, err := f.apply(so)
	if err != nil {
		return syscall.InvalidHandle, err
	}
	s, so.Err = syscall.Socket(family, sotype, proto)
	if err = af.apply(so); err != nil {
		if so.Err == nil {
			syscall.Closesocket(s)
		}
		return syscall.InvalidHandle, err
	}

	sw.smu.Lock()
	defer sw.smu.Unlock()
	if so.Err != nil {
		sw.stats.getLocked(so.Cookie).OpenFailed++
		return syscall.InvalidHandle, so.Err
	}
	nso := sw.addLocked(s, family, sotype, proto)
	sw.stats.getLocked(nso.Cookie).Opened++
	return s, nil
}

// Closesocket wraps syscall.Closesocket.
func (sw *Switch) Closesocket(s syscall.Handle) (err error) {
	so := sw.sockso(s)
	if so == nil {
		return syscall.Closesocket(s)
	}
	sw.fmu.RLock()
	f, _ := sw.fltab[FilterClose]
	sw.fmu.RUnlock()

	af, err := f.apply(so)
	if err != nil {
		return err
	}
	so.Err = syscall.Closesocket(s)
	if err = af.apply(so); err != nil {
		return err
	}

	sw.smu.Lock()
	defer sw.smu.Unlock()
	if so.Err != nil {
		sw.stats.getLocked(so.Cookie).CloseFailed++
		return so.Err
	}
	delete(sw.sotab, s)
	sw.stats.getLocked(so.Cookie).Closed++
	return nil
}

// Conenct wraps syscall.Connect.
func (sw *Switch) Connect(s syscall.Handle, sa syscall.Sockaddr) (err error) {
	so := sw.sockso(s)
	if so == nil {
		return syscall.Connect(s, sa)
	}
	sw.fmu.RLock()
	f, _ := sw.fltab[FilterConnect]
	sw.fmu.RUnlock()

	af, err := f.apply(so)
	if err != nil {
		return err
	}
	so.Err = syscall.Connect(s, sa)
	if err = af.apply(so); err != nil {
		return err
	}

	sw.smu.Lock()
	defer sw.smu.Unlock()
	if so.Err != nil {
		sw.stats.getLocked(so.Cookie).ConnectFailed++
		return so.Err
	}
	sw.stats.getLocked(so.Cookie).Connected++
	return nil
}

// ConenctEx wraps syscall.ConnectEx.
func (sw *Switch) ConnectEx(s syscall.Handle, sa syscall.Sockaddr, b *byte, n uint32, nwr *uint32, o *syscall.Overlapped) (err error) {
	so := sw.sockso(s)
	if so == nil {
		return syscall.ConnectEx(s, sa, b, n, nwr, o)
	}
	sw.fmu.RLock()
	f, _ := sw.fltab[FilterConnect]
	sw.fmu.RUnlock()

	af, err := f.apply(so)
	if err != nil {
		return err
	}
	so.Err = syscall.ConnectEx(s, sa, b, n, nwr, o)
	if err = af.apply(so); err != nil {
		return err
	}

	sw.smu.Lock()
	defer sw.smu.Unlock()
	if so.Err != nil {
		sw.stats.getLocked(so.Cookie).ConnectFailed++
		return so.Err
	}
	sw.stats.getLocked(so.Cookie).Connected++
	return nil
}

// Listen wraps syscall.Listen.
func (sw *Switch) Listen(s syscall.Handle, backlog int) (err error) {
	so := sw.sockso(s)
	if so == nil {
		return syscall.Listen(s, backlog)
	}
	sw.fmu.RLock()
	f, _ := sw.fltab[FilterListen]
	sw.fmu.RUnlock()

	af, err := f.apply(so)
	if err != nil {
		return err
	}
	so.Err = syscall.Listen(s, backlog)
	if err = af.apply(so); err != nil {
		return err
	}

	sw.smu.Lock()
	defer sw.smu.Unlock()
	if so.Err != nil {
		sw.stats.getLocked(so.Cookie).ListenFailed++
		return so.Err
	}
	sw.stats.getLocked(so.Cookie).Listened++
	return nil
}
