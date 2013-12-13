// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"fmt"
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/oracle/serial"
)

// definition reports the location of the definition of an identifier.
//
// TODO(adonovan): opt: for intra-file references, the parser's
// resolution might be enough; we should start with that.
//
func definition(o *Oracle, qpos *QueryPos) (queryResult, error) {
	id, _ := qpos.path[0].(*ast.Ident)
	if id == nil {
		return nil, fmt.Errorf("no identifier here")
	}

	obj := qpos.info.ObjectOf(id)
	if obj == nil {
		// Happens for y in "switch y := x.(type)", but I think that's all.
		return nil, fmt.Errorf("no object for identifier")
	}

	return &definitionResult{qpos, obj}, nil
}

type definitionResult struct {
	qpos *QueryPos
	obj  types.Object // object it denotes
}

func (r *definitionResult) display(printf printfFunc) {
	printf(r.obj, "defined here as %s", r.qpos.ObjectString(r.obj))
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
