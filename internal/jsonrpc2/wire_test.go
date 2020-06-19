// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"

	"golang.org/x/tools/internal/jsonrpc2"
)

var wireIDTestData = []struct {
	name    string
	id      jsonrpc2.ID
	encoded []byte
	plain   string
	quoted  string
}{
	{
		name:    `empty`,
		encoded: []byte(`0`),
		plain:   `0`,
		quoted:  `#0`,
	}, {
		name:    `number`,
		id:      jsonrpc2.NewIntID(43),
		encoded: []byte(`43`),
		plain:   `43`,
		quoted:  `#43`,
	}, {
		name:    `string`,
		id:      jsonrpc2.NewStringID("life"),
		encoded: []byte(`"life"`),
		plain:   `life`,
		quoted:  `"life"`,
	},
}

func TestIDFormat(t *testing.T) {
	for _, test := range wireIDTestData {
		t.Run(test.name, func(t *testing.T) {
			if got := fmt.Sprint(test.id); got != test.plain {
				t.Errorf("got %s expected %s", got, test.plain)
			}
			if got := fmt.Sprintf("%q", test.id); got != test.quoted {
				t.Errorf("got %s want %s", got, test.quoted)
			}
		})
	}
}

func TestIDEncode(t *testing.T) {
	for _, test := range wireIDTestData {
		t.Run(test.name, func(t *testing.T) {
			data, err := json.Marshal(&test.id)
			if err != nil {
				t.Fatal(err)
			}
			checkJSON(t, data, test.encoded)
		})
	}
}

func TestIDDecode(t *testing.T) {
	for _, test := range wireIDTestData {
		t.Run(test.name, func(t *testing.T) {
			var got *jsonrpc2.ID
			if err := json.Unmarshal(test.encoded, &got); err != nil {
				t.Fatal(err)
			}
			if got == nil {
				t.Errorf("got nil want %s", test.id)
			} else if *got != test.id {
				t.Errorf("got %s want %s", got, test.id)
			}
		})
	}
}

func TestErrorResponse(t *testing.T) {
	// originally reported in #39719, this checks that result is not present if
	// it is an error response
	r, _ := jsonrpc2.NewResponse(jsonrpc2.NewIntID(3), nil, fmt.Errorf("computing fix edits"))
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	checkJSON(t, data, []byte(`{
		"jsonrpc":"2.0",
		"error":{
			"code":0,
			"message":"computing fix edits",
			"data":null
		},
		"id":3
	}`))
}

func checkJSON(t *testing.T, got, want []byte) {
	// compare the compact form, to allow for formatting differences
	g := &bytes.Buffer{}
	if err := json.Compact(g, []byte(got)); err != nil {
		t.Fatal(err)
	}
	w := &bytes.Buffer{}
	if err := json.Compact(w, []byte(want)); err != nil {
		t.Fatal(err)
	}
	if g.String() != w.String() {
		t.Fatalf("Got:\n%s\nWant:\n%s", g, w)
	}
}
