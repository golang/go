// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"io"
	"testing"
)

func FuzzUnmarshalJSON(f *testing.F) {
	f.Add([]byte(`{
"object": {
	"slice": [
		1,
		2.0,
		"3",
		[4],
		{5: {}}
	]
},
"slice": [[]],
"string": ":)",
"int": 1e5,
"float": 3e-9"
}`))

	f.Fuzz(func { t, b ->
		for _, typ := range []func() interface{}{
			func() interface{} { return new(interface{}) },
			func() interface{} { return new(map[string]interface{}) },
			func() interface{} { return new([]interface{}) },
		} {
			i := typ()
			if err := Unmarshal(b, i); err != nil {
				return
			}

			encoded, err := Marshal(i)
			if err != nil {
				t.Fatalf("failed to marshal: %s", err)
			}

			if err := Unmarshal(encoded, i); err != nil {
				t.Fatalf("failed to roundtrip: %s", err)
			}
		}
	})
}

func FuzzDecoderToken(f *testing.F) {
	f.Add([]byte(`{
"object": {
	"slice": [
		1,
		2.0,
		"3",
		[4],
		{5: {}}
	]
},
"slice": [[]],
"string": ":)",
"int": 1e5,
"float": 3e-9"
}`))

	f.Fuzz(func { t, b ->
		r := bytes.NewReader(b)
		d := NewDecoder(r)
		for {
			_, err := d.Token()
			if err != nil {
				if err == io.EOF {
					break
				}
				return
			}
		}
	})
}
