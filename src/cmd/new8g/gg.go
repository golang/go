// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "cmd/internal/obj/i386"
import "cmd/internal/gc"

// TODO(rsc):
//	assume CLD?

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// foptoas flags
const (
	Frev  = 1 << 0
	Fpop  = 1 << 1
	Fpop2 = 1 << 2
)

var reg [i386.MAXREG]uint8

var panicdiv *gc.Node

/*
 * cgen.c
 */

/*
 * list.c
 */
