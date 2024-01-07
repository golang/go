// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socktest

import (
	"internal/syscall/windows"
	"syscall"
)

// WSASocket wraps syscall.WSASocket.
func (sw *Switch) WSASocket(family, sotype, proto int32, protinfo *syscall.WSAProtocolInfo, group uint32, flags uint32) (s syscall.Handle, err error) {
	sw.once.Do(sw.init)

	so := &Status{Cookie: cookie(int(family), int(sotype), int(proto))}
	sw.fmu.RLock()
	f, _ := sw.fltab[FilterSocket]
	sw.fmu.RUnlock()

	af, err := f.apply(so)
	if err != nil {
		return syscall.InvalidHandle, err
	}
	s, so.Err = windows.WSASocket(family, sotype, proto, protinfo, group, flags)
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
	nso := sw.addLocked(s, int(family), int(sotype), int(proto))
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

// Connect wraps syscall.Connect.
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

// ConnectEx wraps syscall.ConnectEx.
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

// AcceptEx wraps syscall.AcceptEx.
func (sw *Switch) AcceptEx(ls syscall.Handle, as syscall.Handle, b *byte, rxdatalen uint32, laddrlen uint32, raddrlen uint32, rcvd *uint32, overlapped *syscall.Overlapped) error {
	so := sw.sockso(ls)
	if so == nil {
		return syscall.AcceptEx(ls, as, b, rxdatalen, laddrlen, raddrlen, rcvd, overlapped)
	}
	sw.fmu.RLock()
	f, _ := sw.fltab[FilterAccept]
	sw.fmu.RUnlock()

	af, err := f.apply(so)
	if err != nil {
		return err
	}
	so.Err = syscall.AcceptEx(ls, as, b, rxdatalen, laddrlen, raddrlen, rcvd, overlapped)
	if err = af.apply(so); err != nil {
		return err
	}

	sw.smu.Lock()
	defer sw.smu.Unlock()
	if so.Err != nil {
		sw.stats.getLocked(so.Cookie).AcceptFailed++
		return so.Err
	}
	nso := sw.addLocked(as, so.Cookie.Family(), so.Cookie.Type(), so.Cookie.Protocol())
	sw.stats.getLocked(nso.Cookie).Accepted++
	return nil
}
