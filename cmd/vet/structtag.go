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
	"strings"
	"unicode"
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
// settings.
func validateStructTag(tag string) error {
	elems := strings.Split(tag, " ")
	for _, elem := range elems {
		if elem == "" {
			continue
		}
		fields := strings.SplitN(elem, ":", 2)
		if len(fields) != 2 {
			return errTagSyntax
		}
		key, value := fields[0], fields[1]
		if len(key) == 0 || len(value) < 2 {
			return errTagSyntax
		}
		// Key must not contain control characters or quotes.
		for _, r := range key {
			if r == '"' || unicode.IsControl(r) {
				return errTagKeySyntax
			}
		}
		if value[0] != '"' || value[len(value)-1] != '"' {
			return errTagValueSyntax
		}
		// Value must be quoted string
		_, err := strconv.Unquote(value)
		if err != nil {
			return errTagValueSyntax
		}
	}
	return nil
}
