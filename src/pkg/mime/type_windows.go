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
	if syscall.RegOpenKeyEx(syscall.HKEY_CLASSES_ROOT, syscall.StringToUTF16Ptr(`\`),
		0, syscall.KEY_READ, &root) != 0 {
		return
	}
	defer syscall.RegCloseKey(root)
	var count uint32
	if syscall.RegQueryInfoKey(root, nil, nil, nil, &count, nil, nil, nil, nil, nil, nil, nil) != 0 {
		return
	}
	var buf [1 << 10]uint16
	for i := uint32(0); i < count; i++ {
		n := uint32(len(buf))
		if syscall.RegEnumKeyEx(root, i, &buf[0], &n, nil, nil, nil, nil) != 0 {
			continue
		}
		ext := syscall.UTF16ToString(buf[:])
		if len(ext) < 2 || ext[0] != '.' { // looking for extensions only
			continue
		}
		var h syscall.Handle
		if syscall.RegOpenKeyEx(
			syscall.HKEY_CLASSES_ROOT, syscall.StringToUTF16Ptr(`\`+ext),
			0, syscall.KEY_READ, &h) != 0 {
			continue
		}
		var typ uint32
		n = uint32(len(buf) * 2) // api expects array of bytes, not uint16
		if syscall.RegQueryValueEx(
			h, syscall.StringToUTF16Ptr("Content Type"),
			nil, &typ, (*byte)(unsafe.Pointer(&buf[0])), &n) != 0 {
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
		".bmp": "image/bmp",
		".png": "image/png",
	}
}
