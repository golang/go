// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package src implements source positions.
package src

// Implementation note: This is a thin abstraction over
// the historic representation of source positions via
// global line numbers. The abstraction will make it
// easier to replace this implementation, eventually.

// A Pos represents a source position.
// The zero value for a Pos is a valid unknown position.
type Pos struct {
	// line is an index into the global line table, which maps
	// the corresponding Pos to a file name and source line number.
	line int32
}

// MakePos creates a new Pos from a line index.
// It requires intimate knowledge of the underlying
// implementation and should be used with caution.
func MakePos(line int32) Pos { return Pos{line} }

func (p Pos) IsKnown() bool     { return p.line != 0 }
func (p Pos) Line() int32       { return p.line }
func (p Pos) Before(q Pos) bool { return p.line < q.line }
func (p Pos) After(q Pos) bool  { return p.line > q.line }
