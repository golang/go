// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go2go

import (
	"fmt"
	"go/types"
	"strings"
	"unicode"
)

// We use Oriya digit zero as a separator.
// Do not use this character in your own identifiers.
const nameSep = '୦'

// We use Oriya digit eight to introduce a special character code.
// Do not use this character in your own identifiers.
const nameIntro = '୮'

var nameCodes = map[rune]int{
	' ':       0,
	'*':       1,
	';':       2,
	',':       3,
	'{':       4,
	'}':       5,
	'[':       6,
	']':       7,
	'(':       8,
	')':       9,
	'.':       10,
	'<':       11,
	'-':       12,
	'/':       13,
	nameSep:   14,
	nameIntro: 15,
}

// instantiatedName returns the name of a newly instantiated function.
func (t *translator) instantiatedName(qid qualifiedIdent, types []types.Type) (string, error) {
	var sb strings.Builder
	fmt.Fprintf(&sb, "instantiate%c", nameSep)
	if qid.pkg != nil {
		fmt.Fprintf(&sb, qid.pkg.Name())
	}
	fmt.Fprintf(&sb, "%c%s", nameSep, qid.ident.Name)
	for _, typ := range types {
		sb.WriteRune(nameSep)
		s := typ.String()

		// We have to uniquely translate s into a valid Go identifier.
		// This is not possible in general but we assume that
		// identifiers will not contain nameSep or nameIntro.
		for _, r := range s {
			if (unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_') && r != nameSep && r != nameIntro {
				sb.WriteRune(r)
			} else {
				code, ok := nameCodes[r]
				if !ok {
					panic(fmt.Sprintf("%s: unexpected type string character %q in %q", t.fset.Position(qid.ident.Pos()), r, s))
				}
				fmt.Fprintf(&sb, "%c%x", nameIntro, code)
			}
		}
	}
	return sb.String(), nil
}

// importableName returns a name that we define in each package, so that
// we have something to import to avoid an unused package error.
func (t *translator) importableName() string {
	return "Importable" + string(nameSep)
}
