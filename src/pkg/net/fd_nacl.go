// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os";
	"syscall";
)

type pollster struct{}

func newpollster() (p *pollster, err os.Error) {
	return nil, os.NewSyscallError("networking", syscall.ENACL)
}

func (p *pollster) AddFD(fd int, mode int, repeat bool) os.Error {
	_, err := newpollster();
	return err;
}

func (p *pollster) StopWaiting(fd int, bits uint) {
}

func (p *pollster) DelFD(fd int, mode int)	{}

func (p *pollster) WaitFD(nsec int64) (fd int, mode int, err os.Error) {
	_, err = newpollster();
	return;
}

func (p *pollster) Close() os.Error	{ return nil }
