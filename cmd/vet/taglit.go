// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for untagged struct literals.

package main

import (
	"flag"
	"go/ast"
	"strings"
)

var compositeWhiteList = flag.Bool("compositewhitelist", true, "use composite white list; for testing only")

// checkUntaggedLiteral checks if a composite literal is a struct literal with
// untagged fields.
func (f *File) checkUntaggedLiteral(c *ast.CompositeLit) {
	if !vet("composites") {
		return
	}

	typ := c.Type
	for {
		if typ1, ok := c.Type.(*ast.ParenExpr); ok {
			typ = typ1
			continue
		}
		break
	}

	switch typ.(type) {
	case *ast.ArrayType:
		return
	case *ast.MapType:
		return
	case *ast.StructType:
		return // a literal struct type does not need to use tags
	case *ast.Ident:
		// A simple type name like t or T does not need tags either,
		// since it is almost certainly declared in the current package.
		// (The exception is names being used via import . "pkg", but
		// those are already breaking the Go 1 compatibility promise,
		// so not reporting potential additional breakage seems okay.)
		return
	}

	// Otherwise the type is a selector like pkg.Name.
	// We only care if pkg.Name is a struct, not if it's a map, array, or slice.
	isStruct, typeString := f.pkg.isStruct(c)
	if !isStruct {
		return
	}

	if typeString == "" { // isStruct doesn't know
		typeString = f.gofmt(typ)
	}

	// It's a struct, or we can't tell it's not a struct because we don't have types.

	// Check if the CompositeLit contains an untagged field.
	allKeyValue := true
	for _, e := range c.Elts {
		if _, ok := e.(*ast.KeyValueExpr); !ok {
			allKeyValue = false
			break
		}
	}
	if allKeyValue {
		return
	}

	// Check that the CompositeLit's type has the form pkg.Typ.
	s, ok := c.Type.(*ast.SelectorExpr)
	if !ok {
		return
	}
	pkg, ok := s.X.(*ast.Ident)
	if !ok {
		return
	}

	// Convert the package name to an import path, and compare to a whitelist.
	path := pkgPath(f, pkg.Name)
	if path == "" {
		f.Badf(c.Pos(), "unresolvable package for %s.%s literal", pkg.Name, s.Sel.Name)
		return
	}
	typeName := path + "." + s.Sel.Name
	if *compositeWhiteList && untaggedLiteralWhitelist[typeName] {
		return
	}

	f.Warn(c.Pos(), typeString+" composite literal uses untagged fields")
}

// pkgPath returns the import path "image/png" for the package name "png".
//
// This is based purely on syntax and convention, and not on the imported
// package's contents. It will be incorrect if a package name differs from the
// leaf element of the import path, or if the package was a dot import.
func pkgPath(f *File, pkgName string) (path string) {
	for _, x := range f.file.Imports {
		s := strings.Trim(x.Path.Value, `"`)
		if x.Name != nil {
			// Catch `import pkgName "foo/bar"`.
			if x.Name.Name == pkgName {
				return s
			}
		} else {
			// Catch `import "pkgName"` or `import "foo/bar/pkgName"`.
			if s == pkgName || strings.HasSuffix(s, "/"+pkgName) {
				return s
			}
		}
	}
	return ""
}

var untaggedLiteralWhitelist = map[string]bool{
	/*
		These types are actually slices. Syntactically, we cannot tell
		whether the Typ in pkg.Typ{1, 2, 3} is a slice or a struct, so we
		whitelist all the standard package library's exported slice types.

		find $GOROOT/src/pkg -type f | grep -v _test.go | xargs grep '^type.*\[\]' | \
			grep -v ' map\[' | sed 's,/[^/]*go.type,,' | sed 's,.*src/pkg/,,' | \
			sed 's, ,.,' |  sed 's, .*,,' | grep -v '\.[a-z]' | \
			sort | awk '{ print "\"" $0 "\": true," }'
	*/
	"crypto/x509/pkix.RDNSequence":                  true,
	"crypto/x509/pkix.RelativeDistinguishedNameSET": true,
	"database/sql.RawBytes":                         true,
	"debug/macho.LoadBytes":                         true,
	"encoding/asn1.ObjectIdentifier":                true,
	"encoding/asn1.RawContent":                      true,
	"encoding/json.RawMessage":                      true,
	"encoding/xml.CharData":                         true,
	"encoding/xml.Comment":                          true,
	"encoding/xml.Directive":                        true,
	"go/scanner.ErrorList":                          true,
	"image/color.Palette":                           true,
	"net.HardwareAddr":                              true,
	"net.IP":                                        true,
	"net.IPMask":                                    true,
	"sort.Float64Slice":                             true,
	"sort.IntSlice":                                 true,
	"sort.StringSlice":                              true,
	"unicode.SpecialCase":                           true,

	// These image and image/color struct types are frozen. We will never add fields to them.
	"image/color.Alpha16": true,
	"image/color.Alpha":   true,
	"image/color.Gray16":  true,
	"image/color.Gray":    true,
	"image/color.NRGBA64": true,
	"image/color.NRGBA":   true,
	"image/color.RGBA64":  true,
	"image/color.RGBA":    true,
	"image/color.YCbCr":   true,
	"image.Point":         true,
	"image.Rectangle":     true,
}
