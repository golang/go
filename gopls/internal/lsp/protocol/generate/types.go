// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package main

import (
	"fmt"
	"sort"
)

// Model contains the parsed version of the spec
type Model struct {
	Version       Metadata        `json:"metaData"`
	Requests      []*Request      `json:"requests"`
	Notifications []*Notification `json:"notifications"`
	Structures    []*Structure    `json:"structures"`
	Enumerations  []*Enumeration  `json:"enumerations"`
	TypeAliases   []*TypeAlias    `json:"typeAliases"`
	Line          int             `json:"line"`
}

// Metadata is information about the version of the spec
type Metadata struct {
	Version string `json:"version"`
	Line    int    `json:"line"`
}

// A Request is the parsed version of an LSP request
type Request struct {
	Documentation       string `json:"documentation"`
	ErrorData           *Type  `json:"errorData"`
	Direction           string `json:"messageDirection"`
	Method              string `json:"method"`
	Params              *Type  `json:"params"`
	PartialResult       *Type  `json:"partialResult"`
	Proposed            bool   `json:"proposed"`
	RegistrationMethod  string `json:"registrationMethod"`
	RegistrationOptions *Type  `json:"registrationOptions"`
	Result              *Type  `json:"result"`
	Since               string `json:"since"`
	Line                int    `json:"line"`
}

// A Notificatin is the parsed version of an LSP notification
type Notification struct {
	Documentation       string `json:"documentation"`
	Direction           string `json:"messageDirection"`
	Method              string `json:"method"`
	Params              *Type  `json:"params"`
	Proposed            bool   `json:"proposed"`
	RegistrationMethod  string `json:"registrationMethod"`
	RegistrationOptions *Type  `json:"registrationOptions"`
	Since               string `json:"since"`
	Line                int    `json:"line"`
}

// A Structure is the parsed version of an LSP structure from the spec
type Structure struct {
	Documentation string     `json:"documentation"`
	Extends       []*Type    `json:"extends"`
	Mixins        []*Type    `json:"mixins"`
	Name          string     `json:"name"`
	Properties    []NameType `json:"properties"`
	Proposed      bool       `json:"proposed"`
	Since         string     `json:"since"`
	Line          int        `json:"line"`
}

// An enumeration is the parsed version of an LSP enumeration from the spec
type Enumeration struct {
	Documentation        string      `json:"documentation"`
	Name                 string      `json:"name"`
	Proposed             bool        `json:"proposed"`
	Since                string      `json:"since"`
	SupportsCustomValues bool        `json:"supportsCustomValues"`
	Type                 *Type       `json:"type"`
	Values               []NameValue `json:"values"`
	Line                 int         `json:"line"`
}

// A TypeAlias is the parsed version of an LSP type alias from the spec
type TypeAlias struct {
	Documentation string `json:"documentation"`
	Deprecated    string `json:"deprecated"`
	Name          string `json:"name"`
	Proposed      bool   `json:"proposed"`
	Since         string `json:"since"`
	Type          *Type  `json:"type"`
	Line          int    `json:"line"`
}

// A NameValue describes an enumeration constant
type NameValue struct {
	Documentation string `json:"documentation"`
	Name          string `json:"name"`
	Proposed      bool   `json:"proposed"`
	Since         string `json:"since"`
	Value         any    `json:"value"` // number or string
	Line          int    `json:"line"`
}

// A Type is the parsed version of an LSP type from the spec,
// or a Type the code constructs
type Type struct {
	Kind    string  `json:"kind"`    // -- which kind goes with which field --
	Items   []*Type `json:"items"`   // "and", "or", "tuple"
	Element *Type   `json:"element"` // "array"
	Name    string  `json:"name"`    // "base", "reference"
	Key     *Type   `json:"key"`     // "map"
	Value   any     `json:"value"`   // "map", "stringLiteral", "literal"
	Line    int     `json:"line"`    // JSON source line
}

// ParsedLiteral is Type.Value when Type.Kind is "literal"
type ParseLiteral struct {
	Properties `json:"properties"`
}

// A NameType represents the name and type of a structure element
type NameType struct {
	Name          string `json:"name"`
	Type          *Type  `json:"type"`
	Optional      bool   `json:"optional"`
	Documentation string `json:"documentation"`
	Deprecated    string `json:"deprecated"`
	Since         string `json:"since"`
	Proposed      bool   `json:"proposed"`
	Line          int    `json:"line"`
}

// Properties are the collection of structure fields
type Properties []NameType

// addLineNumbers adds a "line" field to each object in the JSON.
func addLineNumbers(buf []byte) []byte {
	var ans []byte
	// In the specification .json file, the delimiter '{' is
	// always followed by a newline. There are other {s embedded in strings.
	// json.Token does not return \n, or :, or , so using it would
	// require parsing the json to reconstruct the missing information.
	for linecnt, i := 1, 0; i < len(buf); i++ {
		ans = append(ans, buf[i])
		switch buf[i] {
		case '{':
			if buf[i+1] == '\n' {
				ans = append(ans, fmt.Sprintf(`"line": %d, `, linecnt)...)
				// warning: this would fail if the spec file had
				// `"value": {\n}`, but it does not, as comma is a separator.
			}
		case '\n':
			linecnt++
		}
	}
	return ans
}

type sortedMap[T any] map[string]T

func (s sortedMap[T]) keys() []string {
	var keys []string
	for k := range s {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
