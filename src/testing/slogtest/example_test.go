// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slogtest_test

import (
	"bytes"
	"encoding/json"
	"log"
	"log/slog"
	"testing/slogtest"
)

// This example demonstrates one technique for testing a handler with this
// package. The handler is given a [bytes.Buffer] to write to, and each line
// of the resulting output is parsed.
// For JSON output, [encoding/json.Unmarshal] produces a result in the desired
// format when given a pointer to a map[string]any.
func Example_parsing() {
	var buf bytes.Buffer
	h := slog.NewJSONHandler(&buf, nil)

	results := func {
		var ms []map[string]any
		for _, line := range bytes.Split(buf.Bytes(), []byte{'\n'}) {
			if len(line) == 0 {
				continue
			}
			var m map[string]any
			if err := json.Unmarshal(line, &m); err != nil {
				panic(err) // In a real test, use t.Fatal.
			}
			ms = append(ms, m)
		}
		return ms
	}
	err := slogtest.TestHandler(h, results)
	if err != nil {
		log.Fatal(err)
	}

	// Output:
}
