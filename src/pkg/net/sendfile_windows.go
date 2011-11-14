// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"os"
	"syscall"
)

type sendfileOp struct {
	anOp
	src syscall.Handle // source
	n   uint32
}

func (o *sendfileOp) Submit() (err error) {
	return syscall.TransmitFile(o.fd.sysfd, o.src, o.n, 0, &o.o, nil, syscall.TF_WRITE_BEHIND)
}

func (o *sendfileOp) Name() string {
	return "TransmitFile"
}

// sendFile copies the contents of r to c using the TransmitFile
// system call to minimize copies.
//
// if handled == true, sendFile returns the number of bytes copied and any
// non-EOF error.
//
// if handled == false, sendFile performed no work.
//
// Note that sendfile for windows does not suppport >2GB file.
func sendFile(c *netFD, r io.Reader) (written int64, err error, handled bool) {
	var n int64 = 0 // by default, copy until EOF

	lr, ok := r.(*io.LimitedReader)
	if ok {
		n, r = lr.N, lr.R
		if n <= 0 {
			return 0, nil, true
		}
	}
	f, ok := r.(*os.File)
	if !ok {
		return 0, nil, false
	}

	c.wio.Lock()
	defer c.wio.Unlock()
	c.incref()
	defer c.decref()

	var o sendfileOp
	o.Init(c, 'w')
	o.n = uint32(n)
	o.src = f.Fd()
	done, err := iosrv.ExecIO(&o, 0)
	if err != nil {
		return 0, err, false
	}
	if lr != nil {
		lr.N -= int64(done)
	}
	return int64(done), nil, true
}
