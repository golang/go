// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "cmd/internal/obj/arm"

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

const (
	REGALLOC_R0   = arm.REG_R0
	REGALLOC_RMAX = arm.REGEXT
	REGALLOC_F0   = arm.REG_F0
	REGALLOC_FMAX = arm.FREGEXT
)

var reg [REGALLOC_FMAX + 1]uint8

/*
 * cgen
 */

/*
 * list.c
 */

/*
 * reg.c
 */
