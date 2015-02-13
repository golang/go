// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

// Garbage collector liveness bitmap generation.

// The command line flag -live causes this code to print debug information.
// The levels are:
//
//	-live (aka -live=1): print liveness lists as code warnings at safe points
//	-live=2: print an assembly listing with liveness annotations
//	-live=3: print information during each computation phase (much chattier)
//
// Each level includes the earlier output as well.

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used by cmd/gc.

const (
	GcBits          = 4
	BitsPerPointer  = 2
	BitsDead        = 0
	BitsScalar      = 1
	BitsPointer     = 2
	BitsMask        = 3
	PointersPerByte = 8 / BitsPerPointer
	InsData         = 1 + iota - 7
	InsArray
	InsArrayEnd
	InsEnd
	MaxGCMask = 65536
)
