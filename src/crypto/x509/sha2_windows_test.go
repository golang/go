// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import "syscall"

func init() {
	v, err := syscall.GetVersion()
	if err != nil {
		return
	}
	if major := byte(v); major < 6 {
		// Windows XP SP2 and Windows 2003 do not support SHA2.
		// http://blogs.technet.com/b/pki/archive/2010/09/30/sha2-and-windows.aspx
		supportSHA2 = false
	}
}
