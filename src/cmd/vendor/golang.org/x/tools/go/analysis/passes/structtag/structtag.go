// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package structtag defines an Analyzer that checks struct field tags
// are well formed.
package structtag

import (
	"errors"
	"go/ast"
	"go/token"
	"go/types"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = `check that struct field tags conform to reflect.StructTag.Get

Also report certain struct tags (json, xml) used with unexported fields.`

var Analyzer = &analysis.Analyzer{
	Name:             "structtag",
	Doc:              Doc,
	Requires:         []*analysis.Analyzer{inspect.Analyzer},
	RunDespiteErrors: true,
	Run:              run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.StructType)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		styp, ok := pass.TypesInfo.Types[n.(*ast.StructType)].Type.(*types.Struct)
		// Type information may be incomplete.
		if !ok {
			return
		}
		var seen namesSeen
		for i := 0; i < styp.NumFields(); i++ {
			field := styp.Field(i)
			tag := styp.Tag(i)
			checkCanonicalFieldTag(pass, field, tag, &seen)
		}
	})
	return nil, nil
}

// namesSeen keeps track of encoding tags by their key, name, and nested level
// from the initial struct. The level is taken into account because equal
// encoding key names only conflict when at the same level; otherwise, the lower
// level shadows the higher level.
type namesSeen map[uniqueName]token.Pos

type uniqueName struct {
	key   string // "xml" or "json"
	name  string // the encoding name
	level int    // anonymous struct nesting level
}

func (s *namesSeen) Get(key, name string, level int) (token.Pos, bool) {
	if *s == nil {
		*s = make(map[uniqueName]token.Pos)
	}
	pos, ok := (*s)[uniqueName{key, name, level}]
	return pos, ok
}

func (s *namesSeen) Set(key, name string, level int, pos token.Pos) {
	if *s == nil {
		*s = make(map[uniqueName]token.Pos)
	}
	(*s)[uniqueName{key, name, level}] = pos
}

var checkTagDups = []string{"json", "xml"}
var checkTagSpaces = map[string]bool{"json": true, "xml": true, "asn1": true}

// checkCanonicalFieldTag checks a single struct field tag.
func checkCanonicalFieldTag(pass *analysis.Pass, field *types.Var, tag string, seen *namesSeen) {
	switch pass.Pkg.Path() {
	case "encoding/json", "encoding/xml":
		// These packages know how to use their own APIs.
		// Sometimes they are testing what happens to incorrect programs.
		return
	}

	for _, key := range checkTagDups {
		checkTagDuplicates(pass, tag, key, field, field, seen, 1)
	}

	if err := validateStructTag(tag); err != nil {
		pass.Reportf(field.Pos(), "struct field tag %#q not compatible with reflect.StructTag.Get: %s", tag, err)
	}

	// Check for use of json or xml tags with unexported fields.

	// Embedded struct. Nothing to do for now, but that
	// may change, depending on what happens with issue 7363.
	// TODO(adonovan): investigate, now that that issue is fixed.
	if field.Anonymous() {
		return
	}

	if field.Exported() {
		return
	}

	for _, enc := range [...]string{"json", "xml"} {
		switch reflect.StructTag(tag).Get(enc) {
		// Ignore warning if the field not exported and the tag is marked as
		// ignored.
		case "", "-":
		default:
			pass.Reportf(field.Pos(), "struct field %s has %s tag but is not exported", field.Name(), enc)
			return
		}
	}
}

// checkTagDuplicates checks a single struct field tag to see if any tags are
// duplicated. nearest is the field that's closest to the field being checked,
// while still being part of the top-level struct type.
func checkTagDuplicates(pass *analysis.Pass, tag, key string, nearest, field *types.Var, seen *namesSeen, level int) {
	val := reflect.StructTag(tag).Get(key)
	if val == "-" {
		// Ignored, even if the field is anonymous.
		return
	}
	if val == "" || val[0] == ',' {
		if !field.Anonymous() {
			// Ignored if the field isn't anonymous.
			return
		}
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
			checkTagDuplicates(pass, tag, key, nearest, field, seen, level+1)
		}
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
	if pos, ok := seen.Get(key, val, level); ok {
		alsoPos := pass.Fset.Position(pos)
		alsoPos.Column = 0

		// Make the "also at" position relative to the current position,
		// to ensure that all warnings are unambiguous and correct. For
		// example, via anonymous struct fields, it's possible for the
		// two fields to be in different packages and directories.
		thisPos := pass.Fset.Position(field.Pos())
		rel, err := filepath.Rel(filepath.Dir(thisPos.Filename), alsoPos.Filename)
		if err != nil {
			// Possibly because the paths are relative; leave the
			// filename alone.
		} else {
			alsoPos.Filename = rel
		}

		pass.Reportf(nearest.Pos(), "struct field %s repeats %s tag %q also at %s", field.Name(), key, val, alsoPos)
	} else {
		seen.Set(key, val, level, field.Pos())
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
// in the canonical format, as defined by reflect.StructTag.
func validateStructTag(tag string) error {
	// This code is based on the StructTag.Get code in package reflect.

	n := 0
	var keys []string
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
		if i+1 >= len(tag) || tag[i] < ' ' || tag[i] == 0x7f {
			return errTagSyntax
		}
		key := tag[:i]
		keys = append(keys, key)
		tag = tag[i:]

		// If we found a space char here - assume that we have a tag with
		// multiple keys.
		if tag[0] == ' ' {
			continue
		}

		// Spaces were filtered above so we assume that here we have
		// only valid tag value started with `:"`.
		if tag[0] != ':' || tag[1] != '"' {
			return errTagValueSyntax
		}

		// Remove the colon leaving tag at the start of the quoted string.
		tag = tag[1:]

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

		wholeValue, err := strconv.Unquote(qvalue)
		if err != nil {
			return errTagValueSyntax
		}

		for _, key := range keys {
			if !checkTagSpaces[key] {
				continue
			}

			value := wholeValue
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

		keys = keys[:0]
	}
	return nil
}
