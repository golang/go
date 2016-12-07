// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package src implements source positions.
package src

// A Pos represents a source position.
// It is an index into the global line table, which
// maps a Pos to a file name and source line number.
type Pos int32
