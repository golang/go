// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socktest

import "syscall"

// Sockets maps a socket descriptor to the status of socket.
type Sockets map[syscall.Handle]Status

func (sw *Switch) sockso(s syscall.Handle) *Status {
	sw.smu.RLock()
	defer sw.smu.RUnlock()
	so, ok := sw.sotab[s]
	if !ok {
		return nil
	}
	return &so
}

// addLocked returns a new Status without locking.
// sw.smu must be held before call.
func (sw *Switch) addLocked(s syscall.Handle, family, sotype, proto int) *Status {
	sw.once.Do(sw.init)
	so := Status{Cookie: cookie(family, sotype, proto)}
	sw.sotab[s] = so
	return &so
}
