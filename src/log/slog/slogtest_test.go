// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"testing"
	"testing/slogtest"
)

func TestSlogtest(t *testing.T) {
	for _, test := range []struct {
		name  string
		new   func(io.Writer) slog.Handler
		parse func([]byte) (map[string]any, error)
	}{
		{"JSON", func(w io.Writer) slog.Handler { return slog.NewJSONHandler(w, nil) }, parseJSON},
		{"Text", func(w io.Writer) slog.Handler { return slog.NewTextHandler(w, nil) }, parseText},
	} {
		t.Run(test.name, func { t ->
			var buf bytes.Buffer
			h := test.new(&buf)
			results := func {
				ms, err := parseLines(buf.Bytes(), test.parse)
				if err != nil {
					t.Fatal(err)
				}
				return ms
			}
			if err := slogtest.TestHandler(h, results); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func parseLines(src []byte, parse func([]byte) (map[string]any, error)) ([]map[string]any, error) {
	var records []map[string]any
	for _, line := range bytes.Split(src, []byte{'\n'}) {
		if len(line) == 0 {
			continue
		}
		m, err := parse(line)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", string(line), err)
		}
		records = append(records, m)
	}
	return records, nil
}

func parseJSON(bs []byte) (map[string]any, error) {
	var m map[string]any
	if err := json.Unmarshal(bs, &m); err != nil {
		return nil, err
	}
	return m, nil
}

// parseText parses the output of a single call to TextHandler.Handle.
// It can parse the output of the tests in this package,
// but it doesn't handle quoted keys or values.
// It doesn't need to handle all cases, because slogtest deliberately
// uses simple inputs so handler writers can focus on testing
// handler behavior, not parsing.
func parseText(bs []byte) (map[string]any, error) {
	top := map[string]any{}
	s := string(bytes.TrimSpace(bs))
	for len(s) > 0 {
		kv, rest, _ := strings.Cut(s, " ") // assumes exactly one space between attrs
		k, value, found := strings.Cut(kv, "=")
		if !found {
			return nil, fmt.Errorf("no '=' in %q", kv)
		}
		keys := strings.Split(k, ".")
		// Populate a tree of maps for a dotted path such as "a.b.c=x".
		m := top
		for _, key := range keys[:len(keys)-1] {
			x, ok := m[key]
			var m2 map[string]any
			if !ok {
				m2 = map[string]any{}
				m[key] = m2
			} else {
				m2, ok = x.(map[string]any)
				if !ok {
					return nil, fmt.Errorf("value for %q in composite key %q is not map[string]any", key, k)

				}
			}
			m = m2
		}
		m[keys[len(keys)-1]] = value
		s = rest
	}
	return top, nil
}
