// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd

#include <machine/sysarch.h>

#ifdef ARM_TP_ADDRESS
// ARM_TP_ADDRESS is (ARM_VECTORS_HIGH + 0x1000) or 0xffff1000
// and is known to runtime.read_tls_fallback. Verify it with
// cpp.
#if ARM_TP_ADDRESS != 0xffff1000
#error Wrong ARM_TP_ADDRESS!
#endif
#endif
