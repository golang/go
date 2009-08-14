// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A package for multi-precision arithmetic.
// It implements the following numeric types:
//
//	W	unsigned single word with limited precision (uintptr)
//	Z	signed integers
//	Q	rational numbers
//
// Operations follow a regular naming scheme: The
// operation name is followed by the type names of
// the operands. Examples:
//
//	AddWW	implements W + W
//	SubZZ	implements Z + Z
//	MulZW	implements Z * W
//
// All operations returning a multi-precision result take the
// result as the first argument; if it is one of the operands
// it may be overwritten (and its memory reused). To enable
// chaining of operations, the result is also returned.
//
package big

// This file is intentionally left without declarations for now. It may
// contain more documentation eventually; otherwise it should be removed.
