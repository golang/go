// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/ast"
	"go/token"

	"golang.org/x/tools/cmd/guru/serial"
	"golang.org/x/tools/go/loader"
)

// definition reports the location of the definition of an identifier.
func definition(q *Query) error {
	// First try the simple resolution done by parser.
	// It only works for intra-file references but it is very fast.
	// (Extending this approach to all the files of the package,
	// resolved using ast.NewPackage, was not worth the effort.)
	{
		qpos, err := fastQueryPos(q.Build, q.Pos)
		if err != nil {
			return err
		}

		id, _ := qpos.path[0].(*ast.Ident)
		if id == nil {
			return fmt.Errorf("no identifier here")
		}

		if obj := id.Obj; obj != nil && obj.Pos().IsValid() {
			q.Output(qpos.fset, &definitionResult{
				pos:   obj.Pos(),
				descr: fmt.Sprintf("%s %s", obj.Kind, obj.Name),
			})
			return nil // success
		}
	}

	// Run the type checker.
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

	if !obj.Pos().IsValid() {
		return fmt.Errorf("%s is built in", obj.Name())
	}

	q.Output(lprog.Fset, &definitionResult{
		pos:   obj.Pos(),
		descr: qpos.objectString(obj),
	})
	return nil
}

type definitionResult struct {
	pos   token.Pos // (nonzero) location of definition
	descr string    // description of object it denotes
}

func (r *definitionResult) PrintPlain(printf printfFunc) {
	printf(r.pos, "defined here as %s", r.descr)
}

func (r *definitionResult) JSON(fset *token.FileSet) []byte {
	return toJSON(&serial.Definition{
		Desc:   r.descr,
		ObjPos: fset.Position(r.pos).String(),
	})
}
