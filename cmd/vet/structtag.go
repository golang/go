// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for canonical struct tags.

package main

import (
	"go/ast"
	"reflect"
	"strconv"
)

// checkField checks a struct field tag.
func (f *File) checkCanonicalFieldTag(field *ast.Field) {
	if !vet("structtags") {
		return
	}
	if field.Tag == nil {
		return
	}

	tag, err := strconv.Unquote(field.Tag.Value)
	if err != nil {
		f.Badf(field.Pos(), "unable to read struct tag %s", field.Tag.Value)
		return
	}

	// Check tag for validity by appending
	// new key:value to end and checking that
	// the tag parsing code can find it.
	if reflect.StructTag(tag+` _gofix:"_magic"`).Get("_gofix") != "_magic" {
		f.Badf(field.Pos(), "struct field tag %s not compatible with reflect.StructTag.Get", field.Tag.Value)
		return
	}
}
