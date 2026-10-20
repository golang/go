// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"reflect"
	"slices"
	"strconv"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// serializedUnsizedInt is portability::serialized_unsized_int (SPEC Sec 1.2,
// #182): a field of type int or uint in a struct that has a field with a json
// tag. The generated binary layout (M11-S3) writes every integer as 8 bytes
// and ignores tag options, so a field is flagged whatever its own tag says and
// whether or not it is exported, but int and uint are 4 bytes on a 32-bit
// target. A defined type over int, a pointer, and a slice are not flagged: they
// are other types. The rule applies to loGo/s packages only.
var serializedUnsizedInt = Rule{
	ID:    "portability::serialized_unsized_int",
	Group: Portability,
	Check: checkSerializedUnsizedInt,
}

// checkSerializedUnsizedInt is the Check of serializedUnsizedInt.
func checkSerializedUnsizedInt(pass *Pass) {
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			if s, ok := n.(*syntax.StructType); ok {
				reportUnsizedFields(pass, s)
			}
			return true
		})
	}
}

// reportUnsizedFields reports each named int or uint field of s when any field
// of s carries a json tag.
func reportUnsizedFields(pass *Pass, s *syntax.StructType) {
	tagged := slices.ContainsFunc(s.TagList, hasJSONTag)
	for _, f := range s.FieldList {
		name := unsizedFieldName(pass, f)
		if tagged && name != "" {
			pass.Reportf(f.Name, "field %s has type %s, which is 4 bytes on a 32-bit target but 8 bytes in the binary layout; use int32, int64, uint32, or uint64", f.Name.Value, name)
		}
	}
}

// unsizedFieldName returns the name of the type of f, "int" or "uint", when f
// is a named field of that type, and "" for any other field. The type comes
// from the field object the checker recorded, because a loGo/s field default
// wraps the type expression of the field in the syntax tree.
func unsizedFieldName(pass *Pass, f *syntax.Field) string {
	if f.Name == nil {
		return ""
	}
	field, ok := pass.Info.Defs[f.Name].(*types2.Var)
	if !ok {
		return ""
	}
	basic, ok := types2.Unalias(field.Type()).(*types2.Basic)
	if !ok || (basic.Kind() != types2.Int && basic.Kind() != types2.Uint) {
		return ""
	}
	return basic.Name()
}

// hasJSONTag reports whether the struct tag literal tag, which may be nil,
// has a json key.
func hasJSONTag(tag *syntax.BasicLit) bool {
	if tag == nil {
		return false
	}
	value, err := strconv.Unquote(tag.Value)
	if err != nil {
		return false
	}
	_, ok := reflect.StructTag(value).Lookup("json")
	return ok
}
