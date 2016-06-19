// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for canonical struct tags.

package main

import (
	"errors"
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

	if err := validateStructTag(tag); err != nil {
		f.Badf(field.Pos(), "struct field tag %s not compatible with reflect.StructTag.Get: %s", field.Tag.Value, err)
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

	st := reflect.StructTag(tag)
	for _, enc := range [...]string{"json", "xml"} {
		if st.Get(enc) != "" {
			f.Badf(field.Pos(), "struct field %s has %s tag but is not exported", field.Names[0].Name, enc)
			return
		}
	}
}

var (
	errTagSyntax      = errors.New("bad syntax for struct tag pair")
	errTagKeySyntax   = errors.New("bad syntax for struct tag key")
	errTagValueSyntax = errors.New("bad syntax for struct tag value")
)

// validateStructTag parses the struct tag and returns an error if it is not
// in the canonical format, which is a space-separated list of key:"value"
// settings. The value may contain spaces.
func validateStructTag(tag string) error {
	// This code is based on the StructTag.Get code in package reflect.

	for tag != "" {
		// Skip leading space.
		i := 0
		for i < len(tag) && tag[i] == ' ' {
			i++
		}
		tag = tag[i:]
		if tag == "" {
			break
		}

		// Scan to colon. A space, a quote or a control character is a syntax error.
		// Strictly speaking, control chars include the range [0x7f, 0x9f], not just
		// [0x00, 0x1f], but in practice, we ignore the multi-byte control characters
		// as it is simpler to inspect the tag's bytes than the tag's runes.
		i = 0
		for i < len(tag) && tag[i] > ' ' && tag[i] != ':' && tag[i] != '"' && tag[i] != 0x7f {
			i++
		}
		if i == 0 {
			return errTagKeySyntax
		}
		if i+1 >= len(tag) || tag[i] != ':' {
			return errTagSyntax
		}
		if tag[i+1] != '"' {
			return errTagValueSyntax
		}
		tag = tag[i+1:]

		// Scan quoted string to find value.
		i = 1
		for i < len(tag) && tag[i] != '"' {
			if tag[i] == '\\' {
				i++
			}
			i++
		}
		if i >= len(tag) {
			return errTagValueSyntax
		}
		qvalue := tag[:i+1]
		tag = tag[i+1:]

		if _, err := strconv.Unquote(qvalue); err != nil {
			return errTagValueSyntax
		}
	}
	return nil
}
