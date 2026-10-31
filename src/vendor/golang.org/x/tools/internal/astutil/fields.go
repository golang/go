// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"go/ast"
	"iter"
)

// FlatFields 'flattens' an ast.FieldList, returning an iterator over each
// (name, field) combination in the list. For unnamed fields, the identifier is
// nil.
func FlatFields(list *ast.FieldList) iter.Seq2[*ast.Ident, *ast.Field] {
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
