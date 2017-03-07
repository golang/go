// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package src

// Scope represents a lexical scope.
type Scope struct {
	Start, End XPos
	Parent     int32
}

// Scopes represents a tree of lexical scopes
type Scopes struct {
	Scopes   []Scope // lexical scopes
	curscope int32   // current scope index (during noding)
}

// Open starts a new scope, scopes.Curscope is the parent.
func (scopes *Scopes) Open(posTable *PosTable, pos Pos) {
	scope := Scope{Parent: scopes.curscope, Start: NoXPos, End: NoXPos}
	if pos.IsKnown() {
		scope.Start = posTable.XPos(pos)
	}
	scopes.Scopes = append(scopes.Scopes, scope)
	scopes.curscope = int32(len(scopes.Scopes) - 1)
}

// Close ends the current scope.
func (scopes *Scopes) Close(posTable *PosTable, pos Pos) {
	scope := &scopes.Scopes[scopes.curscope]
	if pos.IsKnown() {
		scope.End = posTable.XPos(pos)
	}
	scopes.curscope = scope.Parent
}
