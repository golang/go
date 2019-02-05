// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls for amd64, Windows are implemented in runtime/syscall_windows.goc
//

TEXT ·getprocaddress(SB), 7, $0-32
	JMP	syscall·getprocaddress(SB)

TEXT ·loadlibrary(SB), 7, $0-24
	JMP	syscall·loadlibrary(SB)
