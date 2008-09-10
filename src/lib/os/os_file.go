// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import syscall "syscall"
import os "os"

export func Open(name string, mode int64, flags int64) (ret int64, errno int64) {
	var buf [512]byte;
	if !StringToBytes(&buf, name) {
		return -1, syscall.ENAMETOOLONG
	}
	r, e := syscall.open(&buf[0], mode, flags);  // BUG: should be able to just return
	return r, e
}

export func Close(fd int64) (ret int64, errno int64) {
	r, e := syscall.close(fd);  // BUG: should be able to just return
	return r, e
}

export func Read(fd int64, b *[]byte) (ret int64, errno int64) {
	r, e := syscall.read(fd, &b[0], int64(len(b)));  // BUG: should be able to just return
	return r, e
}

export func Write(fd int64, b *[]byte) (ret int64, errno int64) {
	r, e := syscall.write(fd, &b[0], int64(len(b)));  // BUG: should be able to just return
	return r, e
}

export func WriteString(fd int64, s string) (ret int64, errno int64) {
	b := new([]byte, len(s)+1);
	if !StringToBytes(b, s) {
		return -1, syscall.EIO
	}
	r, e := syscall.write(fd, &b[0], int64(len(s)));  // BUG: should be able to just return
	return r, e
}

