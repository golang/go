// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows
// +build !go1.3

// copied from pkg/runtime
typedef	unsigned int	uint32;
typedef	unsigned long long int	uint64;
#ifdef _64BIT
typedef	uint64		uintptr;
#else
typedef	uint32		uintptr;
#endif

// from sys_386.s or sys_amd64.s
void ·servicemain(void);

void
·getServiceMain(uintptr *r)
{
	*r = (uintptr)·servicemain;
}
