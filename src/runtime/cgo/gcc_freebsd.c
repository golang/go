// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd && (386 || arm || arm64 || riscv64)

#include <machine/sysarch.h>
#include <string.h>
#include "libcgo.h"
#include "libcgo_unix.h"

#ifdef ARM_TP_ADDRESS
// ARM_TP_ADDRESS is (ARM_VECTORS_HIGH + 0x1000) or 0xffff1000
// and is known to runtime.read_tls_fallback. Verify it with
// cpp.
#if ARM_TP_ADDRESS != 0xffff1000
#error Wrong ARM_TP_ADDRESS!
#endif
#endif

static void (*setg_gcc)(void*);

void
x_cgo_init(G *g, void (*setg)(void*))
{
	setg_gcc = setg;
	_cgo_set_stacklo(g, NULL);
}

extern void crosscall1(void (*fn)(void), void (*setg_gcc)(void*), void *g);
void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	free(v);

	crosscall1(ts.fn, setg_gcc, ts.g);
	return nil;
}
