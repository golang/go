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

func init() {
	register("structtags",
		"check that struct field tags have canonical format and apply to exported fields as needed",
		checkCanonicalFieldTag,
		field)
}

// checkCanonicalFieldTag checks a struct field tag.
func checkCanonicalFieldTag(f *File, node ast.Node) {
	field := node.(*ast.Field)
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
	st := reflect.StructTag(tag + ` _gofix:"_magic"`)
	if st.Get("_gofix") != "_magic" {
		f.Badf(field.Pos(), "struct field tag %s not compatible with reflect.StructTag.Get", field.Tag.Value)
		return
	}

	// Check for use of json or xml tags with unexported fields.

	// Embedded struct. Nothing to do for now, but that
	// may change, depending on what happens with issue 7363.
	if len(field.Names) == 0 {
		return
	}

	if field.Names[0].IsExported() {
		return
	}

	for _, enc := range [...]string{"json", "xml"} {
		if st.Get(enc) != "" {
			f.Badf(field.Pos(), "struct field %s has %s tag but is not exported", field.Names[0].Name, enc)
			return
		}
	}
}
