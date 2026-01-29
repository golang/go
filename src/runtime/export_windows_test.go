// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export guts for testing.

package runtime

import (
	"internal/runtime/syscall/windows"
	"unsafe"
)

var (
	OsYield                 = osyield
	TimeBeginPeriodRetValue = &timeBeginPeriodRetValue
)

func NumberOfProcessors() int32 {
	var info windows.SystemInfo
	stdcall(_GetSystemInfo, uintptr(unsafe.Pointer(&info)))
	return int32(info.NumberOfProcessors)
}

func GetCallerFp() uintptr {
	return getcallerfp()
}
