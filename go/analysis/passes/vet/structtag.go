// +build ignore

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for canonical struct tags.

package main

import (
	"errors"
	"go/ast"
	"go/token"
	"go/types"
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
	astType := node.(*ast.StructType)
	typ := f.pkg.types[astType].Type.(*types.Struct)
	var seen map[[2]string]token.Pos
	for i := 0; i < typ.NumFields(); i++ {
		field := typ.Field(i)
		tag := typ.Tag(i)
		checkCanonicalFieldTag(f, astType, field, tag, &seen)
	}
}

var checkTagDups = []string{"json", "xml"}
var checkTagSpaces = map[string]bool{"json": true, "xml": true, "asn1": true}

// checkCanonicalFieldTag checks a single struct field tag.
// top is the top-level struct type that is currently being checked.
func checkCanonicalFieldTag(f *File, top *ast.StructType, field *types.Var, tag string, seen *map[[2]string]token.Pos) {
	for _, key := range checkTagDups {
		checkTagDuplicates(f, tag, key, field, field, seen)
	}

	if err := validateStructTag(tag); err != nil {
		f.Badf(field.Pos(), "struct field tag %#q not compatible with reflect.StructTag.Get: %s", tag, err)
	}

	// Check for use of json or xml tags with unexported fields.

	// Embedded struct. Nothing to do for now, but that
	// may change, depending on what happens with issue 7363.
	if field.Anonymous() {
		return
	}

	if field.Exported() {
		return
	}

	for _, enc := range [...]string{"json", "xml"} {
		if reflect.StructTag(tag).Get(enc) != "" {
			f.Badf(field.Pos(), "struct field %s has %s tag but is not exported", field.Name(), enc)
			return
		}
	}
}

// checkTagDuplicates checks a single struct field tag to see if any tags are
// duplicated. nearest is the field that's closest to the field being checked,
// while still being part of the top-level struct type.
func checkTagDuplicates(f *File, tag, key string, nearest, field *types.Var, seen *map[[2]string]token.Pos) {
	val := reflect.StructTag(tag).Get(key)
	if val == "-" {
		// Ignored, even if the field is anonymous.
		return
	}
	if val == "" || val[0] == ',' {
		if field.Anonymous() {
			typ, ok := field.Type().Underlying().(*types.Struct)
			if !ok {
				return
			}
			for i := 0; i < typ.NumFields(); i++ {
				field := typ.Field(i)
				if !field.Exported() {
					continue
				}
				tag := typ.Tag(i)
				checkTagDuplicates(f, tag, key, nearest, field, seen)
			}
		}
		// Ignored if the field isn't anonymous.
		return
	}
	if key == "xml" && field.Name() == "XMLName" {
		// XMLName defines the XML element name of the struct being
		// checked. That name cannot collide with element or attribute
		// names defined on other fields of the struct. Vet does not have a
		// check for untagged fields of type struct defining their own name
		// by containing a field named XMLName; see issue 18256.
		return
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
		f.Badf(nearest.Pos(), "struct field %s repeats %s tag %q also at %s", field.Name(), key, val, f.loc(pos))
	} else {
		(*seen)[[2]string{key, val}] = field.Pos()
	}
}

var (
	errTagSyntax      = errors.New("bad syntax for struct tag pair")
	errTagKeySyntax   = errors.New("bad syntax for struct tag key")
	errTagValueSyntax = errors.New("bad syntax for struct tag value")
	errTagValueSpace  = errors.New("suspicious space in struct tag value")
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
		key := tag[:i]
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

		value, err := strconv.Unquote(qvalue)
		if err != nil {
			return errTagValueSyntax
		}

		if !checkTagSpaces[key] {
			continue
		}

		switch key {
		case "xml":
			// If the first or last character in the XML tag is a space, it is
			// suspicious.
			if strings.Trim(value, " ") != value {
				return errTagValueSpace
			}

			// If there are multiple spaces, they are suspicious.
			if strings.Count(value, " ") > 1 {
				return errTagValueSpace
			}

			// If there is no comma, skip the rest of the checks.
			comma := strings.IndexRune(value, ',')
			if comma < 0 {
				continue
			}

			// If the character before a comma is a space, this is suspicious.
			if comma > 0 && value[comma-1] == ' ' {
				return errTagValueSpace
			}
			value = value[comma+1:]
		case "json":
			// JSON allows using spaces in the name, so skip it.
			comma := strings.IndexRune(value, ',')
			if comma < 0 {
				continue
			}
			value = value[comma+1:]
		}

		if strings.IndexByte(value, ' ') >= 0 {
			return errTagValueSpace
		}
	}
	return nil
}
