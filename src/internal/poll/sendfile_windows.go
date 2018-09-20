// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import "syscall"

// SendFile wraps the TransmitFile call.
func SendFile(fd *FD, src syscall.Handle, n int64) (int64, error) {
	ft, err := syscall.GetFileType(src)
	if err != nil {
		return 0, err
	}
	// TransmitFile does not work with pipes
	if ft == syscall.FILE_TYPE_PIPE {
		return 0, syscall.ESPIPE
	}

	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()

	o := &fd.wop
	o.qty = uint32(n)
	o.handle = src

	// TODO(brainman): skip calling syscall.Seek if OS allows it
	curpos, err := syscall.Seek(o.handle, 0, 1)
	if err != nil {
		return 0, err
	}

	o.o.Offset = uint32(curpos)
	o.o.OffsetHigh = uint32(curpos >> 32)

	done, err := wsrv.ExecIO(o, func(o *operation) error {
		return syscall.TransmitFile(o.fd.Sysfd, o.handle, o.qty, 0, &o.o, nil, syscall.TF_WRITE_BEHIND)
	})
	if err == nil {
		// Some versions of Windows (Windows 10 1803) do not set
		// file position after TransmitFile completes.
		// So just use Seek to set file position.
		_, err = syscall.Seek(o.handle, curpos+int64(done), 0)
	}
	return int64(done), err
}
