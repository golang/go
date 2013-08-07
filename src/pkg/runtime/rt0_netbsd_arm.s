// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../../cmd/ld/textflag.h"

// FreeBSD/NetBSD and Linux use the same linkage to main

TEXT _rt0_arm_netbsd(SB),NOSPLIT,$-4
	B _rt0_go(SB)
