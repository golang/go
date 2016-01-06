// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.5

package oracle

import (
	"fmt"
	"go/ast"
	"go/token"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/types"
	"golang.org/x/tools/oracle/serial"
)

// definition reports the location of the definition of an identifier.
//
// TODO(adonovan): opt: for intra-file references, the parser's
// resolution might be enough; we should start with that.
//
func definition(q *Query) error {
	lconf := loader.Config{Build: q.Build}
	allowErrors(&lconf)

	if _, err := importQueryPackage(q.Pos, &lconf); err != nil {
		return err
	}

	// Load/parse/type-check the program.
	lprog, err := lconf.Load()
	if err != nil {
		return err
	}
	q.Fset = lprog.Fset

	qpos, err := parseQueryPos(lprog, q.Pos, false)
	if err != nil {
		return err
	}

	id, _ := qpos.path[0].(*ast.Ident)
	if id == nil {
		return fmt.Errorf("no identifier here")
	}

	obj := qpos.info.ObjectOf(id)
	if obj == nil {
		// Happens for y in "switch y := x.(type)",
		// and the package declaration,
		// but I think that's all.
		return fmt.Errorf("no object for identifier")
	}

	q.result = &definitionResult{qpos, obj}
	return nil
}

type definitionResult struct {
	qpos *queryPos
	obj  types.Object // object it denotes
}

func (r *definitionResult) display(printf printfFunc) {
	printf(r.obj, "defined here as %s", r.qpos.objectString(r.obj))
}

func (r *definitionResult) toSerial(res *serial.Result, fset *token.FileSet) {
	definition := &serial.Definition{
		Desc: r.obj.String(),
	}
	if pos := r.obj.Pos(); pos != token.NoPos { // Package objects have no Pos()
		definition.ObjPos = fset.Position(pos).String()
	}
	res.Definition = definition
}
