// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

#include "textflag.h"

#define SYNC	WORD $0xf

TEXT ·publicationBarrier(SB),NOSPLIT,$-8-0
	SYNC
	RET
