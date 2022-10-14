// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

// a spec contains the specification of the protocol, and derived information.
type spec struct {
	model *Model

	// combined Requests and Notifications, indexed by method (e.g., "textDocument/didOpen")
	byMethod sortedMap[Message]

	// Structures, Enumerations, and TypeAliases, indexed by name used in
	// the .json specification file
	// (Some Structure and Enumeration names need to be changed for Go,
	// such as _Initialize)
	byName sortedMap[Defined]

	// computed type information
	nameToTypes sortedMap[[]*Type] // all the uses of a type name

	// remember which types are in a union type
	orTypes sortedMap[sortedMap[bool]]

	// information about the version of vscode-languageclient-node
	githash string
	modTime time.Time
}

// parse the specification file and return a spec.
// (TestParseContents checks that the parse gets all the fields of the specification)
func parse(dir string) *spec {
	fname := filepath.Join(dir, "protocol", "metaModel.json")
	buf, err := os.ReadFile(fname)
	if err != nil {
		log.Fatalf("could not read metaModel.json: %v", err)
	}
	// line numbers in the .json file occur as comments in tsprotocol.go
	newbuf := addLineNumbers(buf)
	var v Model
	if err := json.Unmarshal(newbuf, &v); err != nil {
		log.Fatalf("could not unmarshal metaModel.json: %v", err)
	}

	ans := &spec{
		model:       &v,
		byMethod:    make(sortedMap[Message]),
		byName:      make(sortedMap[Defined]),
		nameToTypes: make(sortedMap[[]*Type]),
		orTypes:     make(sortedMap[sortedMap[bool]]),
	}
	ans.githash, ans.modTime = gitInfo(dir)
	return ans
}

// gitInfo returns the git hash and modtime of the repository.
func gitInfo(dir string) (string, time.Time) {
	fname := dir + "/.git/HEAD"
	buf, err := os.ReadFile(fname)
	if err != nil {
		log.Fatal(err)
	}
	buf = bytes.TrimSpace(buf)
	var githash string
	if len(buf) == 40 {
		githash = string(buf[:40])
	} else if bytes.HasPrefix(buf, []byte("ref: ")) {
		fname = dir + "/.git/" + string(buf[5:])
		buf, err = os.ReadFile(fname)
		if err != nil {
			log.Fatal(err)
		}
		githash = string(buf[:40])
	} else {
		log.Fatalf("githash cannot be recovered from %s", fname)
	}
	loadTime := time.Now()
	return githash, loadTime
}

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

// Type.Value has to be treated specially for literals and maps
func (t *Type) UnmarshalJSON(data []byte) error {
	// First unmarshal only the unambiguous fields.
	var x struct {
		Kind    string  `json:"kind"`
		Items   []*Type `json:"items"`
		Element *Type   `json:"element"`
		Name    string  `json:"name"`
		Key     *Type   `json:"key"`
		Value   any     `json:"value"`
		Line    int     `json:"line"`
	}
	if err := json.Unmarshal(data, &x); err != nil {
		return err
	}
	*t = Type{
		Kind:    x.Kind,
		Items:   x.Items,
		Element: x.Element,
		Name:    x.Name,
		Value:   x.Value,
		Line:    x.Line,
	}

	// Then unmarshal the 'value' field based on the kind.
	// This depends on Unmarshal ignoring fields it doesn't know about.
	switch x.Kind {
	case "map":
		var x struct {
			Key   *Type `json:"key"`
			Value *Type `json:"value"`
		}
		if err := json.Unmarshal(data, &x); err != nil {
			return fmt.Errorf("Type.kind=map: %v", err)
		}
		t.Key = x.Key
		t.Value = x.Value

	case "literal":
		var z struct {
			Value ParseLiteral `json:"value"`
		}

		if err := json.Unmarshal(data, &z); err != nil {
			return fmt.Errorf("Type.kind=literal: %v", err)
		}
		t.Value = z.Value

	case "base", "reference", "array", "and", "or", "tuple",
		"stringLiteral":
		// nop. never seen integerLiteral or booleanLiteral.

	default:
		return fmt.Errorf("cannot decode Type.kind %q: %s", x.Kind, data)
	}
	return nil
}
