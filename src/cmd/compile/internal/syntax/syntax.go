// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"io"
	"os"
)

// Mode describes the parser mode.
type Mode uint

// Modes supported by the parser.
const (
	CheckBranches Mode = 1 << iota // check correct use of labels, break, continue, and goto statements
)

// Error describes a syntax error. Error implements the error interface.
type Error struct {
	Pos Pos
	Msg string
}

func (err Error) Error() string {
	return fmt.Sprintf("%s: %s", err.Pos, err.Msg)
}

var _ error = Error{} // verify that Error implements error

// An ErrorHandler is called for each error encountered reading a .go file.
type ErrorHandler func(err error)

// A Pragma value augments a package, import, const, func, type, or var declaration.
// Its meaning is entirely up to the PragmaHandler,
// except that nil is used to mean “no pragma seen.”
type Pragma interface{}

// A PragmaHandler is used to process //go: directives while scanning.
// It is passed the current pragma value, which starts out being nil,
// and it returns an updated pragma value.
// The text is the directive, with the "//" prefix stripped.
// The current pragma is saved at each package, import, const, func, type, or var
// declaration, into the File, ImportDecl, ConstDecl, FuncDecl, TypeDecl, or VarDecl node.
//
// If text is the empty string, the pragma is being returned
// to the handler unused, meaning it appeared before a non-declaration.
// The handler may wish to report an error. In this case, pos is the
// current parser position, not the position of the pragma itself.
// Blank specifies whether the line is blank before the pragma.
type PragmaHandler func(pos Pos, blank bool, text string, current Pragma) Pragma

// Parse parses a single Go source file from src and returns the corresponding
// syntax tree. If there are errors, Parse will return the first error found,
// and a possibly partially constructed syntax tree, or nil.
//
// If errh != nil, it is called with each error encountered, and Parse will
// process as much source as possible. In this case, the returned syntax tree
// is only nil if no correct package clause was found.
// If errh is nil, Parse will terminate immediately upon encountering the first
// error, and the returned syntax tree is nil.
//
// If pragh != nil, it is called with each pragma encountered.
func Parse(base *PosBase, src io.Reader, errh ErrorHandler, pragh PragmaHandler, mode Mode) (_ *File, first error) {
	defer func() {
		if p := recover(); p != nil {
			if err, ok := p.(Error); ok {
				first = err
				return
			}
			panic(p)
		}
	}()

	var p parser
	p.init(base, src, errh, pragh, mode)
	p.next()
	return p.fileOrNil(), p.first
}

// ParseFile behaves like Parse but it reads the source from the named file.
func ParseFile(filename string, errh ErrorHandler, pragh PragmaHandler, mode Mode) (*File, error) {
	f, err := os.Open(filename)
	if err != nil {
		if errh != nil {
			errh(err)
		}
		return nil, err
	}
	defer f.Close()
	return Parse(NewFileBase(filename), f, errh, pragh, mode)
}
