// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package whitelist defines exceptions for the vet tool.
package whitelist

// UnkeyedLiteral are types that are actually slices, but
// syntactically, we cannot tell whether the Typ in pkg.Typ{1, 2, 3}
// is a slice or a struct, so we whitelist all the standard package
// library's exported slice types.
var UnkeyedLiteral = map[string]bool{
	/*
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
	"image.Uniform":       true,
}
