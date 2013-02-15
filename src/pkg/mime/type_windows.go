// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"syscall"
	"unsafe"
)

func initMime() {
	var root syscall.Handle
	rootpathp, _ := syscall.UTF16PtrFromString(`\`)
	if syscall.RegOpenKeyEx(syscall.HKEY_CLASSES_ROOT, rootpathp,
		0, syscall.KEY_READ, &root) != nil {
		return
	}
	defer syscall.RegCloseKey(root)
	var count uint32
	if syscall.RegQueryInfoKey(root, nil, nil, nil, &count, nil, nil, nil, nil, nil, nil, nil) != nil {
		return
	}
	var buf [1 << 10]uint16
	for i := uint32(0); i < count; i++ {
		n := uint32(len(buf))
		if syscall.RegEnumKeyEx(root, i, &buf[0], &n, nil, nil, nil, nil) != nil {
			continue
		}
		ext := syscall.UTF16ToString(buf[:])
		if len(ext) < 2 || ext[0] != '.' { // looking for extensions only
			continue
		}
		var h syscall.Handle
		extpathp, _ := syscall.UTF16PtrFromString(`\` + ext)
		if syscall.RegOpenKeyEx(
			syscall.HKEY_CLASSES_ROOT, extpathp,
			0, syscall.KEY_READ, &h) != nil {
			continue
		}
		var typ uint32
		n = uint32(len(buf) * 2) // api expects array of bytes, not uint16
		contenttypep, _ := syscall.UTF16PtrFromString("Content Type")
		if syscall.RegQueryValueEx(
			h, contenttypep,
			nil, &typ, (*byte)(unsafe.Pointer(&buf[0])), &n) != nil {
			syscall.RegCloseKey(h)
			continue
		}
		syscall.RegCloseKey(h)
		if typ != syscall.REG_SZ { // null terminated strings only
			continue
		}
		mimeType := syscall.UTF16ToString(buf[:])
		setExtensionType(ext, mimeType)
	}
}

func initMimeForTests() map[string]string {
	return map[string]string{
		".png": "image/png",
	}
}
