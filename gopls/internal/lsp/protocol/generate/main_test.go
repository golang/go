// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"testing"
)

// These tests require the result of
//"git clone https://github.com/microsoft/vscode-languageserver-node" in the HOME directory

// this is not a test, but a way to get code coverage,
// (in vscode, just run the test with  "go.coverOnSingleTest": true)
func TestAll(t *testing.T) {
	t.Skip("needs vscode-languageserver-node repository")
	log.SetFlags(log.Lshortfile)
	main()
}

// check that the parsed file includes all the information
// from the json file. This test will fail if the spec
// introduces new fields. (one can test this test by
// commenting out the version field in Model.)
func TestParseContents(t *testing.T) {
	t.Skip("needs vscode-languageserver-node repository")
	log.SetFlags(log.Lshortfile)

	// compute our parse of the specification
	dir := os.Getenv("HOME") + "/vscode-languageserver-node"
	fname := dir + "/protocol/metaModel.json"
	v := parse(fname)
	out, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var our interface{}
	if err := json.Unmarshal(out, &our); err != nil {
		t.Fatal(err)
	}

	// process the json file
	buf, err := os.ReadFile(fname)
	if err != nil {
		t.Fatalf("could not read metaModel.json: %v", err)
	}
	var raw interface{}
	if err := json.Unmarshal(buf, &raw); err != nil {
		t.Fatal(err)
	}

	// convert to strings showing the fields
	them := flatten(raw)
	us := flatten(our)

	// everything in them should be in us
	lesser := make(sortedMap[bool])
	for _, s := range them {
		lesser[s] = true
	}
	greater := make(sortedMap[bool]) // set of fields we have
	for _, s := range us {
		greater[s] = true
	}
	for _, k := range lesser.keys() { // set if fields they have
		if !greater[k] {
			t.Errorf("missing %s", k)
		}
	}
}

// flatten(nil) = "nil"
// flatten(v string) = fmt.Sprintf("%q", v)
// flatten(v float64)= fmt.Sprintf("%g", v)
// flatten(v bool) = fmt.Sprintf("%v", v)
// flatten(v []any) = []string{"[0]"flatten(v[0]), "[1]"flatten(v[1]), ...}
// flatten(v map[string]any) = {"key1": flatten(v["key1"]), "key2": flatten(v["key2"]), ...}
func flatten(x any) []string {
	switch v := x.(type) {
	case nil:
		return []string{"nil"}
	case string:
		return []string{fmt.Sprintf("%q", v)}
	case float64:
		return []string{fmt.Sprintf("%g", v)}
	case bool:
		return []string{fmt.Sprintf("%v", v)}
	case []any:
		var ans []string
		for i, x := range v {
			idx := fmt.Sprintf("[%.3d]", i)
			for _, s := range flatten(x) {
				ans = append(ans, idx+s)
			}
		}
		return ans
	case map[string]any:
		var ans []string
		for k, x := range v {
			idx := fmt.Sprintf("%q:", k)
			for _, s := range flatten(x) {
				ans = append(ans, idx+s)
			}
		}
		return ans
	default:
		log.Fatalf("unexpected type %T", x)
		return nil
	}
}
