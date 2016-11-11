// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"os"
	"sync/atomic"
	"syscall"
	"time"
)

type atomicBool int32

func (b *atomicBool) isSet() bool { return atomic.LoadInt32((*int32)(b)) != 0 }
func (b *atomicBool) setFalse()   { atomic.StoreInt32((*int32)(b), 0) }
func (b *atomicBool) setTrue()    { atomic.StoreInt32((*int32)(b), 1) }

// Network file descriptor.
type netFD struct {
	// locking/lifetime of sysfd + serialize access to Read and Write methods
	fdmu fdMutex

	// immutable until Close
	net               string
	n                 string
	dir               string
	listen, ctl, data *os.File
	laddr, raddr      Addr
	isStream          bool

	// deadlines
	raio      *asyncIO
	waio      *asyncIO
	rtimer    *time.Timer
	wtimer    *time.Timer
	rtimedout atomicBool // set true when read deadline has been reached
	wtimedout atomicBool // set true when write deadline has been reached
}

var (
	netdir string // default network
)

func sysInit() {
	netdir = "/net"
}

func newFD(net, name string, listen, ctl, data *os.File, laddr, raddr Addr) (*netFD, error) {
	return &netFD{
		net:    net,
		n:      name,
		dir:    netdir + "/" + net + "/" + name,
		listen: listen,
		ctl:    ctl, data: data,
		laddr: laddr,
		raddr: raddr,
	}, nil
}

func (fd *netFD) init() error {
	// stub for future fd.pd.Init(fd)
	return nil
}

func (fd *netFD) name() string {
	var ls, rs string
	if fd.laddr != nil {
		ls = fd.laddr.String()
	}
	if fd.raddr != nil {
		rs = fd.raddr.String()
	}
	return fd.net + ":" + ls + "->" + rs
}

func (fd *netFD) ok() bool { return fd != nil && fd.ctl != nil }

func (fd *netFD) destroy() {
	if !fd.ok() {
		return
	}
	err := fd.ctl.Close()
	if fd.data != nil {
		if err1 := fd.data.Close(); err1 != nil && err == nil {
			err = err1
		}
	}
	if fd.listen != nil {
		if err1 := fd.listen.Close(); err1 != nil && err == nil {
			err = err1
		}
	}
	fd.ctl = nil
	fd.data = nil
	fd.listen = nil
}

func (fd *netFD) Read(b []byte) (n int, err error) {
	if fd.rtimedout.isSet() {
		return 0, errTimeout
	}
	if !fd.ok() || fd.data == nil {
		return 0, syscall.EINVAL
	}
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()
	if len(b) == 0 {
		return 0, nil
	}
	fd.raio = newAsyncIO(fd.data.Read, b)
	n, err = fd.raio.Wait()
	fd.raio = nil
	if isHangup(err) {
		err = io.EOF
	}
	if isInterrupted(err) {
		err = errTimeout
	}
	if fd.net == "udp" && err == io.EOF {
		n = 0
		err = nil
	}
	return
}

func (fd *netFD) Write(b []byte) (n int, err error) {
	if fd.wtimedout.isSet() {
		return 0, errTimeout
	}
	if !fd.ok() || fd.data == nil {
		return 0, syscall.EINVAL
	}
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	fd.waio = newAsyncIO(fd.data.Write, b)
	n, err = fd.waio.Wait()
	fd.waio = nil
	if isInterrupted(err) {
		err = errTimeout
	}
	return
}

func (fd *netFD) closeRead() error {
	if !fd.ok() {
		return syscall.EINVAL
	}
	return syscall.EPLAN9
}

func (fd *netFD) closeWrite() error {
	if !fd.ok() {
		return syscall.EINVAL
	}
	return syscall.EPLAN9
}

func (fd *netFD) Close() error {
	if !fd.fdmu.increfAndClose() {
		return errClosing
	}
	if !fd.ok() {
		return syscall.EINVAL
	}
	if fd.net == "tcp" {
		// The following line is required to unblock Reads.
		_, err := fd.ctl.WriteString("close")
		if err != nil {
			return err
		}
	}
	err := fd.ctl.Close()
	if fd.data != nil {
		if err1 := fd.data.Close(); err1 != nil && err == nil {
			err = err1
		}
	}
	if fd.listen != nil {
		if err1 := fd.listen.Close(); err1 != nil && err == nil {
			err = err1
		}
	}
	fd.ctl = nil
	fd.data = nil
	fd.listen = nil
	return err
}

// This method is only called via Conn.
func (fd *netFD) dup() (*os.File, error) {
	if !fd.ok() || fd.data == nil {
		return nil, syscall.EINVAL
	}
	return fd.file(fd.data, fd.dir+"/data")
}

func (l *TCPListener) dup() (*os.File, error) {
	if !l.fd.ok() {
		return nil, syscall.EINVAL
	}
	return l.fd.file(l.fd.ctl, l.fd.dir+"/ctl")
}

func (fd *netFD) file(f *os.File, s string) (*os.File, error) {
	dfd, err := syscall.Dup(int(f.Fd()), -1)
	if err != nil {
		return nil, os.NewSyscallError("dup", err)
	}
	return os.NewFile(uintptr(dfd), s), nil
}

func (fd *netFD) setDeadline(t time.Time) error {
	return setDeadlineImpl(fd, t, 'r'+'w')
}

func (fd *netFD) setReadDeadline(t time.Time) error {
	return setDeadlineImpl(fd, t, 'r')
}

func (fd *netFD) setWriteDeadline(t time.Time) error {
	return setDeadlineImpl(fd, t, 'w')
}

func setDeadlineImpl(fd *netFD, t time.Time, mode int) error {
	d := t.Sub(time.Now())
	if mode == 'r' || mode == 'r'+'w' {
		fd.rtimedout.setFalse()
	}
	if mode == 'w' || mode == 'r'+'w' {
		fd.wtimedout.setFalse()
	}
	if t.IsZero() || d < 0 {
		// Stop timer
		if mode == 'r' || mode == 'r'+'w' {
			if fd.rtimer != nil {
				fd.rtimer.Stop()
			}
			fd.rtimer = nil
		}
		if mode == 'w' || mode == 'r'+'w' {
			if fd.wtimer != nil {
				fd.wtimer.Stop()
			}
			fd.wtimer = nil
		}
	} else {
		// Interrupt I/O operation once timer has expired
		if mode == 'r' || mode == 'r'+'w' {
			fd.rtimer = time.AfterFunc(d, func() {
				fd.rtimedout.setTrue()
				if fd.raio != nil {
					fd.raio.Cancel()
				}
			})
		}
		if mode == 'w' || mode == 'r'+'w' {
			fd.wtimer = time.AfterFunc(d, func() {
				fd.wtimedout.setTrue()
				if fd.waio != nil {
					fd.waio.Cancel()
				}
			})
		}
	}
	if !t.IsZero() && d < 0 {
		// Interrupt current I/O operation
		if mode == 'r' || mode == 'r'+'w' {
			fd.rtimedout.setTrue()
			if fd.raio != nil {
				fd.raio.Cancel()
			}
		}
		if mode == 'w' || mode == 'r'+'w' {
			fd.wtimedout.setTrue()
			if fd.waio != nil {
				fd.waio.Cancel()
			}
		}
	}
	return nil
}

func setReadBuffer(fd *netFD, bytes int) error {
	return syscall.EPLAN9
}

func setWriteBuffer(fd *netFD, bytes int) error {
	return syscall.EPLAN9
}

func isHangup(err error) bool {
	return err != nil && stringsHasSuffix(err.Error(), "Hangup")
}

func isInterrupted(err error) bool {
	return err != nil && stringsHasSuffix(err.Error(), "interrupted")
}
