// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
)

const symbolSetup = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

import (
	"encoding/json"
	"fmt"
)

func main() { // function
	fmt.Println("Hello")
}

var myvar int // variable

type myType string // basic type

type myDecoder json.Decoder // to use the encoding/json import

func (m *myType) Blahblah() {} // method

type myStruct struct { // struct type
	myStructField int // struct field
}

type myInterface interface { // interface
	DoSomeCoolStuff() string // interface method
}

type embed struct {
	myStruct

	nestedStruct struct {
		nestedField int

		nestedStruct2 struct {
			int
		}
	}

	nestedInterface interface {
		myInterface
		nestedMethod()
	}
}
-- p/p.go --
package p

const Message = "Hello World." // constant
`

var caseSensitiveSymbolChecks = map[string]*expSymbolInformation{
	"main": {
		Name: pString("main.main"),
		Kind: pKind(protocol.Function),
		Location: &expLocation{
			Path: pString("main.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(7),
					Column: pInt(5),
				},
			},
		},
	},
	"Message": {
		Name: pString("p.Message"),
		Kind: pKind(protocol.Constant),
		Location: &expLocation{
			Path: pString("p/p.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(2),
					Column: pInt(6),
				},
			},
		},
	},
	"myvar": {
		Name: pString("main.myvar"),
		Kind: pKind(protocol.Variable),
		Location: &expLocation{
			Path: pString("main.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(11),
					Column: pInt(4),
				},
			},
		},
	},
	"myType": {
		Name: pString("main.myType"),
		Kind: pKind(protocol.String),
		Location: &expLocation{
			Path: pString("main.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(13),
					Column: pInt(5),
				},
			},
		},
	},
	"Blahblah": {
		Name: pString("main.myType.Blahblah"),
		Kind: pKind(protocol.Method),
		Location: &expLocation{
			Path: pString("main.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(17),
					Column: pInt(17),
				},
			},
		},
	},
	"NewEncoder": {
		Name: pString("json.NewEncoder"),
		Kind: pKind(protocol.Function),
	},
	"myStruct": {
		Name: pString("main.myStruct"),
		Kind: pKind(protocol.Struct),
		Location: &expLocation{
			Path: pString("main.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(19),
					Column: pInt(5),
				},
			},
		},
	},
	// TODO: not sure we should be returning struct fields
	"myStructField": {
		Name: pString("main.myStruct.myStructField"),
		Kind: pKind(protocol.Field),
		Location: &expLocation{
			Path: pString("main.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(20),
					Column: pInt(1),
				},
			},
		},
	},
	"myInterface": {
		Name: pString("main.myInterface"),
		Kind: pKind(protocol.Interface),
		Location: &expLocation{
			Path: pString("main.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(23),
					Column: pInt(5),
				},
			},
		},
	},
	// TODO: not sure we should be returning interface methods
	"DoSomeCoolStuff": {
		Name: pString("main.myInterface.DoSomeCoolStuff"),
		Kind: pKind(protocol.Method),
		Location: &expLocation{
			Path: pString("main.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(24),
					Column: pInt(1),
				},
			},
		},
	},

	"embed.myStruct": {
		Name: pString("main.embed.myStruct"),
		Kind: pKind(protocol.Field),
		Location: &expLocation{
			Path: pString("main.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(28),
					Column: pInt(1),
				},
			},
		},
	},

	"nestedStruct2.int": {
		Name: pString("main.embed.nestedStruct.nestedStruct2.int"),
		Kind: pKind(protocol.Field),
		Location: &expLocation{
			Path: pString("main.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(34),
					Column: pInt(3),
				},
			},
		},
	},

	"nestedInterface.myInterface": {
		Name: pString("main.embed.nestedInterface.myInterface"),
		Kind: pKind(protocol.Interface),
		Location: &expLocation{
			Path: pString("main.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(39),
					Column: pInt(2),
				},
			},
		},
	},

	"nestedInterface.nestedMethod": {
		Name: pString("main.embed.nestedInterface.nestedMethod"),
		Kind: pKind(protocol.Method),
		Location: &expLocation{
			Path: pString("main.go"),
			Range: &expRange{
				Start: &expPos{
					Line:   pInt(40),
					Column: pInt(2),
				},
			},
		},
	},
}

var caseInsensitiveSymbolChecks = map[string]*expSymbolInformation{
	"Main": caseSensitiveSymbolChecks["main"],
}

var fuzzySymbolChecks = map[string]*expSymbolInformation{
	"Mn": caseSensitiveSymbolChecks["main"],
}

// TestSymbolPos tests that, at a basic level, we get the correct position
// information for symbols matches that are returned.
func TestSymbolPos(t *testing.T) {
	checkChecks(t, "caseSensitive", caseSensitiveSymbolChecks)
	checkChecks(t, "caseInsensitive", caseInsensitiveSymbolChecks)
	checkChecks(t, "fuzzy", fuzzySymbolChecks)
}

func checkChecks(t *testing.T, matcher string, checks map[string]*expSymbolInformation) {
	t.Helper()
	withOptions(
		EditorConfig{SymbolMatcher: &matcher},
	).run(t, symbolSetup, func(t *testing.T, env *Env) {
		t.Run(matcher, func(t *testing.T) {
			for query, exp := range checks {
				t.Run(query, func(t *testing.T) {
					res := env.Symbol(query)
					if !exp.matchAgainst(res) {
						t.Fatalf("failed to find a match against query %q for %v,\ngot: %v", query, exp, res)
					}
				})
			}
		})
	})
}
