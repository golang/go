// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"go/ast"
	"iter"
)

// Fields returns an iterator over the parameters or struct fields
// represented by a FieldList, yielding one (name, field) pair per
// element counted by [ast.FieldList.NumFields]. For each name in a
// Field, it yields that name and the Field; for a Field with no names,
// it yields nil and the Field. Thus a Field with several names is
// yielded once for each name. A nil list yields nothing.
//
// Fields is a template for the proposed method
// (*ast.FieldList).Fields; see https://go.dev/issue/73395.
func Fields(list *ast.FieldList) iter.Seq2[*ast.Ident, *ast.Field] {
	return func(yield func(*ast.Ident, *ast.Field) bool) {
		if list == nil {
			return
		}

		for _, field := range list.List {
			if len(field.Names) == 0 {
				if !yield(nil, field) {
					return
				}
			} else {
				for _, name := range field.Names {
					if !yield(name, field) {
						return
					}
				}
			}
		}
	}
}
