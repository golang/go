// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"encoding/json"
	"fmt"

	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
)

// expSymbolInformation and the types it references are pointer-based versions
// of fake.SymbolInformation, used to make it easier to partially assert
// against values of type fake.SymbolInformation

// expSymbolInformation is a pointer-based version of fake.SymbolInformation
type expSymbolInformation struct {
	Name     *string
	Kind     *protocol.SymbolKind
	Location *expLocation
}

func (e *expSymbolInformation) matchAgainst(sis []fake.SymbolInformation) bool {
	for _, si := range sis {
		if e.match(si) {
			return true
		}
	}
	return false
}

func (e *expSymbolInformation) match(si fake.SymbolInformation) bool {
	if e.Name != nil && *e.Name != si.Name {
		return false
	}
	if e.Kind != nil && *e.Kind != si.Kind {
		return false
	}
	if e.Location != nil && !e.Location.match(si.Location) {
		return false
	}
	return true
}

func (e *expSymbolInformation) String() string {
	byts, err := json.MarshalIndent(e, "", "  ")
	if err != nil {
		panic(fmt.Errorf("failed to json.Marshal *expSymbolInformation: %v", err))
	}
	return string(byts)
}

// expLocation is a pointer-based version of fake.Location
type expLocation struct {
	Path  *string
	Range *expRange
}

func (e *expLocation) match(l fake.Location) bool {
	if e.Path != nil && *e.Path != l.Path {
		return false
	}
	if e.Range != nil && !e.Range.match(l.Range) {
		return false
	}
	return true
}

// expRange is a pointer-based version of fake.Range
type expRange struct {
	Start *expPos
	End   *expPos
}

func (e *expRange) match(l fake.Range) bool {
	if e.Start != nil && !e.Start.match(l.Start) {
		return false
	}
	if e.End != nil && !e.End.match(l.End) {
		return false
	}
	return true
}

// expPos is a pointer-based version of fake.Pos
type expPos struct {
	Line   *int
	Column *int
}

func (e *expPos) match(l fake.Pos) bool {
	if e.Line != nil && *e.Line != l.Line {
		return false
	}
	if e.Column != nil && *e.Column != l.Column {
		return false
	}
	return true
}

func pString(s string) *string {
	return &s
}

func pInt(i int) *int {
	return &i
}

func pKind(k protocol.SymbolKind) *protocol.SymbolKind {
	return &k
}
