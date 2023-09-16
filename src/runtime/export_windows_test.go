// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export guts for testing.

package runtime

import "unsafe"

const MaxArgs = maxArgs

var (
	OsYield                 = osyield
	TimeBeginPeriodRetValue = &timeBeginPeriodRetValue
)

func NumberOfProcessors() int32 {
	var info systeminfo
	stdcall1(_GetSystemInfo, uintptr(unsafe.Pointer(&info)))
	return int32(info.dwnumberofprocessors)
}

type ContextStub struct {
	context
}

func (c ContextStub) GetPC() uintptr {
	return c.ip()
}

func NewContextStub() *ContextStub {
	var ctx context
	ctx.set_ip(getcallerpc())
	ctx.set_sp(getcallersp())
	fp := getfp()
	// getfp is not implemented on windows/386 and windows/arm,
	// in which case it returns 0.
	if fp != 0 {
		ctx.set_fp(*(*uintptr)(unsafe.Pointer(fp)))
	}
	return &ContextStub{ctx}
}
