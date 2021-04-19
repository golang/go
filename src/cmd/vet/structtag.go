// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for canonical struct tags.

package main

import (
	"errors"
	"go/ast"
	"go/token"
	"reflect"
	"strconv"
	"strings"
)

func init() {
	register("structtags",
		"check that struct field tags have canonical format and apply to exported fields as needed",
		checkStructFieldTags,
		structType)
}

// checkStructFieldTags checks all the field tags of a struct, including checking for duplicates.
func checkStructFieldTags(f *File, node ast.Node) {
	var seen map[[2]string]token.Pos
	for _, field := range node.(*ast.StructType).Fields.List {
		checkCanonicalFieldTag(f, field, &seen)
	}
}

var checkTagDups = []string{"json", "xml"}

// checkCanonicalFieldTag checks a single struct field tag.
func checkCanonicalFieldTag(f *File, field *ast.Field, seen *map[[2]string]token.Pos) {
	if field.Tag == nil {
		return
	}

	tag, err := strconv.Unquote(field.Tag.Value)
	if err != nil {
		f.Badf(field.Pos(), "unable to read struct tag %s", field.Tag.Value)
		return
	}

	if err := validateStructTag(tag); err != nil {
		raw, _ := strconv.Unquote(field.Tag.Value) // field.Tag.Value is known to be a quoted string
		f.Badf(field.Pos(), "struct field tag %#q not compatible with reflect.StructTag.Get: %s", raw, err)
	}

	for _, key := range checkTagDups {
		val := reflect.StructTag(tag).Get(key)
		if val == "" || val == "-" || val[0] == ',' {
			continue
		}
		if key == "xml" && len(field.Names) > 0 && field.Names[0].Name == "XMLName" {
			// XMLName defines the XML element name of the struct being
			// checked. That name cannot collide with element or attribute
			// names defined on other fields of the struct. Vet does not have a
			// check for untagged fields of type struct defining their own name
			// by containing a field named XMLName; see issue 18256.
			continue
		}
		if i := strings.Index(val, ","); i >= 0 {
			if key == "xml" {
				// Use a separate namespace for XML attributes.
				for _, opt := range strings.Split(val[i:], ",") {
					if opt == "attr" {
						key += " attribute" // Key is part of the error message.
						break
					}
				}
			}
			val = val[:i]
		}
		if *seen == nil {
			*seen = map[[2]string]token.Pos{}
		}
		if pos, ok := (*seen)[[2]string{key, val}]; ok {
			var name string
			if len(field.Names) > 0 {
				name = field.Names[0].Name
			} else {
				name = field.Type.(*ast.Ident).Name
			}
			f.Badf(field.Pos(), "struct field %s repeats %s tag %q also at %s", name, key, val, f.loc(pos))
		} else {
			(*seen)[[2]string{key, val}] = field.Pos()
		}
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
		if reflect.StructTag(tag).Get(enc) != "" {
			f.Badf(field.Pos(), "struct field %s has %s tag but is not exported", field.Names[0].Name, enc)
			return
		}
	}
}

var (
	errTagSyntax      = errors.New("bad syntax for struct tag pair")
	errTagKeySyntax   = errors.New("bad syntax for struct tag key")
	errTagValueSyntax = errors.New("bad syntax for struct tag value")
	errTagSpace       = errors.New("key:\"value\" pairs not separated by spaces")
)

// validateStructTag parses the struct tag and returns an error if it is not
// in the canonical format, which is a space-separated list of key:"value"
// settings. The value may contain spaces.
func validateStructTag(tag string) error {
	// This code is based on the StructTag.Get code in package reflect.

	n := 0
	for ; tag != ""; n++ {
		if n > 0 && tag != "" && tag[0] != ' ' {
			// More restrictive than reflect, but catches likely mistakes
			// like `x:"foo",y:"bar"`, which parses as `x:"foo" ,y:"bar"` with second key ",y".
			return errTagSpace
		}
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
