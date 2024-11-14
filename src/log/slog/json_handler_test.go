// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog/internal/buffer"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestJSONHandler(t *testing.T) {
	for _, test := range []struct {
		name string
		opts HandlerOptions
		want string
	}{
		{
			"none",
			HandlerOptions{},
			`{"time":"2000-01-02T03:04:05Z","level":"INFO","msg":"m","a":1,"m":{"b":2}}`,
		},
		{
			"replace",
			HandlerOptions{ReplaceAttr: upperCaseKey},
			`{"TIME":"2000-01-02T03:04:05Z","LEVEL":"INFO","MSG":"m","A":1,"M":{"b":2}}`,
		},
	} {
		t.Run(test.name, func { t ->
			var buf bytes.Buffer
			h := NewJSONHandler(&buf, &test.opts)
			r := NewRecord(testTime, LevelInfo, "m", 0)
			r.AddAttrs(Int("a", 1), Any("m", map[string]int{"b": 2}))
			if err := h.Handle(context.Background(), r); err != nil {
				t.Fatal(err)
			}
			got := strings.TrimSuffix(buf.String(), "\n")
			if got != test.want {
				t.Errorf("\ngot  %s\nwant %s", got, test.want)
			}
		})
	}
}

// for testing json.Marshaler
type jsonMarshaler struct {
	s string
}

func (j jsonMarshaler) String() string { return j.s } // should be ignored

func (j jsonMarshaler) MarshalJSON() ([]byte, error) {
	if j.s == "" {
		return nil, errors.New("json: empty string")
	}
	return []byte(fmt.Sprintf(`[%q]`, j.s)), nil
}

type jsonMarshalerError struct {
	jsonMarshaler
}

func (jsonMarshalerError) Error() string { return "oops" }

func TestAppendJSONValue(t *testing.T) {
	// jsonAppendAttrValue should always agree with json.Marshal.
	for _, value := range []any{
		"hello\r\n\t\a",
		`"[{escape}]"`,
		"<escapeHTML&>",
		// \u2028\u2029 is an edge case in JavaScript vs JSON.
		// \xF6 is an incomplete encoding.
		"\u03B8\u2028\u2029\uFFFF\xF6",
		`-123`,
		int64(-9_200_123_456_789_123_456),
		uint64(9_200_123_456_789_123_456),
		-12.75,
		1.23e-9,
		false,
		time.Minute,
		testTime,
		jsonMarshaler{"xyz"},
		jsonMarshalerError{jsonMarshaler{"pqr"}},
		LevelWarn,
	} {
		got := jsonValueString(AnyValue(value))
		want, err := marshalJSON(value)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("%v: got %s, want %s", value, got, want)
		}
	}
}

func marshalJSON(x any) (string, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(x); err != nil {
		return "", err
	}
	return strings.TrimSpace(buf.String()), nil
}

func TestJSONAppendAttrValueSpecial(t *testing.T) {
	// Attr values that render differently from json.Marshal.
	for _, test := range []struct {
		value any
		want  string
	}{
		{math.NaN(), `"!ERROR:json: unsupported value: NaN"`},
		{math.Inf(+1), `"!ERROR:json: unsupported value: +Inf"`},
		{math.Inf(-1), `"!ERROR:json: unsupported value: -Inf"`},
		{io.EOF, `"EOF"`},
	} {
		got := jsonValueString(AnyValue(test.value))
		if got != test.want {
			t.Errorf("%v: got %s, want %s", test.value, got, test.want)
		}
	}
}

func jsonValueString(v Value) string {
	var buf []byte
	s := &handleState{h: &commonHandler{json: true}, buf: (*buffer.Buffer)(&buf)}
	if err := appendJSONValue(s, v); err != nil {
		s.appendError(err)
	}
	return string(buf)
}

func BenchmarkJSONHandler(b *testing.B) {
	for _, bench := range []struct {
		name string
		opts HandlerOptions
	}{
		{"defaults", HandlerOptions{}},
		{"time format", HandlerOptions{
			ReplaceAttr: func(_ []string, a Attr) Attr {
				v := a.Value
				if v.Kind() == KindTime {
					return String(a.Key, v.Time().Format(rfc3339Millis))
				}
				if a.Key == "level" {
					return Attr{"severity", a.Value}
				}
				return a
			},
		}},
		{"time unix", HandlerOptions{
			ReplaceAttr: func(_ []string, a Attr) Attr {
				v := a.Value
				if v.Kind() == KindTime {
					return Int64(a.Key, v.Time().UnixNano())
				}
				if a.Key == "level" {
					return Attr{"severity", a.Value}
				}
				return a
			},
		}},
	} {
		b.Run(bench.name, func { b ->
			ctx := context.Background()
			l := New(NewJSONHandler(io.Discard, &bench.opts)).With(
				String("program", "my-test-program"),
				String("package", "log/slog"),
				String("traceID", "2039232309232309"),
				String("URL", "https://pkg.go.dev/golang.org/x/log/slog"))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				l.LogAttrs(ctx, LevelInfo, "this is a typical log message",
					String("module", "github.com/google/go-cmp"),
					String("version", "v1.23.4"),
					Int("count", 23),
					Int("number", 123456),
				)
			}
		})
	}
}

func BenchmarkPreformatting(b *testing.B) {
	type req struct {
		Method  string
		URL     string
		TraceID string
		Addr    string
	}

	structAttrs := []any{
		String("program", "my-test-program"),
		String("package", "log/slog"),
		Any("request", &req{
			Method:  "GET",
			URL:     "https://pkg.go.dev/golang.org/x/log/slog",
			TraceID: "2039232309232309",
			Addr:    "127.0.0.1:8080",
		}),
	}

	outFile, err := os.Create(filepath.Join(b.TempDir(), "bench.log"))
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := outFile.Close(); err != nil {
			b.Fatal(err)
		}
	}()

	for _, bench := range []struct {
		name  string
		wc    io.Writer
		attrs []any
	}{
		{"separate", io.Discard, []any{
			String("program", "my-test-program"),
			String("package", "log/slog"),
			String("method", "GET"),
			String("URL", "https://pkg.go.dev/golang.org/x/log/slog"),
			String("traceID", "2039232309232309"),
			String("addr", "127.0.0.1:8080"),
		}},
		{"struct", io.Discard, structAttrs},
		{"struct file", outFile, structAttrs},
	} {
		ctx := context.Background()
		b.Run(bench.name, func { b ->
			l := New(NewJSONHandler(bench.wc, nil)).With(bench.attrs...)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				l.LogAttrs(ctx, LevelInfo, "this is a typical log message",
					String("module", "github.com/google/go-cmp"),
					String("version", "v1.23.4"),
					Int("count", 23),
					Int("number", 123456),
				)
			}
		})
	}
}

func BenchmarkJSONEncoding(b *testing.B) {
	value := 3.14
	buf := buffer.New()
	defer buf.Free()
	b.Run("json.Marshal", func { b ->
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			by, err := json.Marshal(value)
			if err != nil {
				b.Fatal(err)
			}
			buf.Write(by)
			*buf = (*buf)[:0]
		}
	})
	b.Run("Encoder.Encode", func { b ->
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if err := json.NewEncoder(buf).Encode(value); err != nil {
				b.Fatal(err)
			}
			*buf = (*buf)[:0]
		}
	})
	_ = buf
}
