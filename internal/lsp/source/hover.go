// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/token"
	"go/types"
)

func Hover(ctx context.Context, f File, pos token.Pos) (string, Range, error) {
	fAST, err := f.GetAST()
	if err != nil {
		return "", Range{}, err
	}
	pkg, err := f.GetPackage()
	if err != nil {
		return "", Range{}, err
	}
	i, err := findIdentifier(fAST, pos)
	if err != nil {
		return "", Range{}, err
	}
	if i.ident == nil {
		return "", Range{}, fmt.Errorf("not a valid identifier")
	}
	obj := pkg.TypesInfo.ObjectOf(i.ident)
	if obj == nil {
		return "", Range{}, fmt.Errorf("no object")
	}
	if i.wasEmbeddedField {
		// the original position was on the embedded field declaration
		// so we try to dig out the type and jump to that instead
		if v, ok := obj.(*types.Var); ok {
			if n, ok := v.Type().(*types.Named); ok {
				obj = n.Obj()
			}
		}
	}
	// TODO(rstambler): Add documentation and improve quality of object string.
	content := types.ObjectString(obj, qualifier(fAST, pkg.Types, pkg.TypesInfo))
	markdown := "```go\n" + content + "\n```"
	return markdown, Range{
		Start: i.ident.Pos(),
		End:   i.ident.End(),
	}, nil
}
