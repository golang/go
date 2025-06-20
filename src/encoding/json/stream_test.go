// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.jsonv2

package json

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"path"
	"reflect"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
)

// TODO(https://go.dev/issue/52751): Replace with native testing support.

// CaseName is a case name annotated with a file and line.
type CaseName struct {
	Name  string
	Where CasePos
}

// Name annotates a case name with the file and line of the caller.
func Name(s string) (c CaseName) {
	c.Name = s
	runtime.Callers(2, c.Where.pc[:])
	return c
}

// CasePos represents a file and line number.
type CasePos struct{ pc [1]uintptr }

func (pos CasePos) String() string {
	frames := runtime.CallersFrames(pos.pc[:])
	frame, _ := frames.Next()
	return fmt.Sprintf("%s:%d", path.Base(frame.File), frame.Line)
}

// Test values for the stream test.
// One of each JSON kind.
var streamTest = []any{
	0.1,
	"hello",
	nil,
	true,
	false,
	[]any{"a", "b", "c"},
	map[string]any{"K": "Kelvin", "ß": "long s"},
	3.14, // another value to make sure something can follow map
}

var streamEncoded = `0.1
"hello"
null
true
false
["a","b","c"]
{"ß":"long s","K":"Kelvin"}
3.14
`

func TestEncoder(t *testing.T) {
	for i := 0; i <= len(streamTest); i++ {
		var buf strings.Builder
		enc := NewEncoder(&buf)
		// Check that enc.SetIndent("", "") turns off indentation.
		enc.SetIndent(">", ".")
		enc.SetIndent("", "")
		for j, v := range streamTest[0:i] {
			if err := enc.Encode(v); err != nil {
				t.Fatalf("#%d.%d Encode error: %v", i, j, err)
			}
		}
		if got, want := buf.String(), nlines(streamEncoded, i); got != want {
			t.Errorf("encoding %d items: mismatch:", i)
			diff(t, []byte(got), []byte(want))
			break
		}
	}
}

func TestEncoderErrorAndReuseEncodeState(t *testing.T) {
	// Disable the GC temporarily to prevent encodeState's in Pool being cleaned away during the test.
	percent := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(percent)

	// Trigger an error in Marshal with cyclic data.
	type Dummy struct {
		Name string
		Next *Dummy
	}
	dummy := Dummy{Name: "Dummy"}
	dummy.Next = &dummy

	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	if err := enc.Encode(dummy); err == nil {
		t.Errorf("Encode(dummy) error: got nil, want non-nil")
	}

	type Data struct {
		A string
		I int
	}
	want := Data{A: "a", I: 1}
	if err := enc.Encode(want); err != nil {
		t.Errorf("Marshal error: %v", err)
	}

	var got Data
	if err := Unmarshal(buf.Bytes(), &got); err != nil {
		t.Errorf("Unmarshal error: %v", err)
	}
	if got != want {
		t.Errorf("Marshal/Unmarshal roundtrip:\n\tgot:  %v\n\twant: %v", got, want)
	}
}

var streamEncodedIndent = `0.1
"hello"
null
true
false
[
>."a",
>."b",
>."c"
>]
{
>."ß": "long s",
>."K": "Kelvin"
>}
3.14
`

func TestEncoderIndent(t *testing.T) {
	var buf strings.Builder
	enc := NewEncoder(&buf)
	enc.SetIndent(">", ".")
	for _, v := range streamTest {
		enc.Encode(v)
	}
	if got, want := buf.String(), streamEncodedIndent; got != want {
		t.Errorf("Encode mismatch:\ngot:\n%s\n\nwant:\n%s", got, want)
		diff(t, []byte(got), []byte(want))
	}
}

type strMarshaler string

func (s strMarshaler) MarshalJSON() ([]byte, error) {
	return []byte(s), nil
}

type strPtrMarshaler string

func (s *strPtrMarshaler) MarshalJSON() ([]byte, error) {
	return []byte(*s), nil
}

func TestEncoderSetEscapeHTML(t *testing.T) {
	var c C
	var ct CText
	var tagStruct struct {
		Valid   int `json:"<>&#! "`
		Invalid int `json:"\\"`
	}

	// This case is particularly interesting, as we force the encoder to
	// take the address of the Ptr field to use its MarshalJSON method. This
	// is why the '&' is important.
	marshalerStruct := &struct {
		NonPtr strMarshaler
		Ptr    strPtrMarshaler
	}{`"<str>"`, `"<str>"`}

	// https://golang.org/issue/34154
	stringOption := struct {
		Bar string `json:"bar,string"`
	}{`<html>foobar</html>`}

	tests := []struct {
		CaseName
		v          any
		wantEscape string
		want       string
	}{
		{Name("c"), c, `"\u003c\u0026\u003e"`, `"<&>"`},
		{Name("ct"), ct, `"\"\u003c\u0026\u003e\""`, `"\"<&>\""`},
		{Name(`"<&>"`), "<&>", `"\u003c\u0026\u003e"`, `"<&>"`},
		{
			Name("tagStruct"), tagStruct,
			`{"\u003c\u003e\u0026#! ":0,"Invalid":0}`,
			`{"<>&#! ":0,"Invalid":0}`,
		},
		{
			Name(`"<str>"`), marshalerStruct,
			`{"NonPtr":"\u003cstr\u003e","Ptr":"\u003cstr\u003e"}`,
			`{"NonPtr":"<str>","Ptr":"<str>"}`,
		},
		{
			Name("stringOption"), stringOption,
			`{"bar":"\"\\u003chtml\\u003efoobar\\u003c/html\\u003e\""}`,
			`{"bar":"\"<html>foobar</html>\""}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			var buf strings.Builder
			enc := NewEncoder(&buf)
			if err := enc.Encode(tt.v); err != nil {
				t.Fatalf("%s: Encode(%s) error: %s", tt.Where, tt.Name, err)
			}
			if got := strings.TrimSpace(buf.String()); got != tt.wantEscape {
				t.Errorf("%s: Encode(%s):\n\tgot:  %s\n\twant: %s", tt.Where, tt.Name, got, tt.wantEscape)
			}
			buf.Reset()
			enc.SetEscapeHTML(false)
			if err := enc.Encode(tt.v); err != nil {
				t.Fatalf("%s: SetEscapeHTML(false) Encode(%s) error: %s", tt.Where, tt.Name, err)
			}
			if got := strings.TrimSpace(buf.String()); got != tt.want {
				t.Errorf("%s: SetEscapeHTML(false) Encode(%s):\n\tgot:  %s\n\twant: %s",
					tt.Where, tt.Name, got, tt.want)
			}
		})
	}
}

func TestDecoder(t *testing.T) {
	for i := 0; i <= len(streamTest); i++ {
		// Use stream without newlines as input,
		// just to stress the decoder even more.
		// Our test input does not include back-to-back numbers.
		// Otherwise stripping the newlines would
		// merge two adjacent JSON values.
		var buf bytes.Buffer
		for _, c := range nlines(streamEncoded, i) {
			if c != '\n' {
				buf.WriteRune(c)
			}
		}
		out := make([]any, i)
		dec := NewDecoder(&buf)
		for j := range out {
			if err := dec.Decode(&out[j]); err != nil {
				t.Fatalf("decode #%d/%d error: %v", j, i, err)
			}
		}
		if !reflect.DeepEqual(out, streamTest[0:i]) {
			t.Errorf("decoding %d items: mismatch:", i)
			for j := range out {
				if !reflect.DeepEqual(out[j], streamTest[j]) {
					t.Errorf("#%d:\n\tgot:  %v\n\twant: %v", j, out[j], streamTest[j])
				}
			}
			break
		}
	}
}

func TestDecoderBuffered(t *testing.T) {
	r := strings.NewReader(`{"Name": "Gopher"} extra `)
	var m struct {
		Name string
	}
	d := NewDecoder(r)
	err := d.Decode(&m)
	if err != nil {
		t.Fatal(err)
	}
	if m.Name != "Gopher" {
		t.Errorf("Name = %s, want Gopher", m.Name)
	}
	rest, err := io.ReadAll(d.Buffered())
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(rest), " extra "; got != want {
		t.Errorf("Remaining = %s, want %s", got, want)
	}
}

func nlines(s string, n int) string {
	if n <= 0 {
		return ""
	}
	for i, c := range s {
		if c == '\n' {
			if n--; n == 0 {
				return s[0 : i+1]
			}
		}
	}
	return s
}

func TestRawMessage(t *testing.T) {
	var data struct {
		X  float64
		Id RawMessage
		Y  float32
	}
	const raw = `["\u0056",null]`
	const want = `{"X":0.1,"Id":["\u0056",null],"Y":0.2}`
	err := Unmarshal([]byte(want), &data)
	if err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if string([]byte(data.Id)) != raw {
		t.Fatalf("Unmarshal:\n\tgot:  %s\n\twant: %s", []byte(data.Id), raw)
	}
	got, err := Marshal(&data)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	if string(got) != want {
		t.Fatalf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

func TestNullRawMessage(t *testing.T) {
	var data struct {
		X     float64
		Id    RawMessage
		IdPtr *RawMessage
		Y     float32
	}
	const want = `{"X":0.1,"Id":null,"IdPtr":null,"Y":0.2}`
	err := Unmarshal([]byte(want), &data)
	if err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if want, got := "null", string(data.Id); want != got {
		t.Fatalf("Unmarshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
	if data.IdPtr != nil {
		t.Fatalf("pointer mismatch: got non-nil, want nil")
	}
	got, err := Marshal(&data)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	if string(got) != want {
		t.Fatalf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

func TestBlocking(t *testing.T) {
	tests := []struct {
		CaseName
		in string
	}{
		{Name(""), `{"x": 1}`},
		{Name(""), `[1, 2, 3]`},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			r, w := net.Pipe()
			go w.Write([]byte(tt.in))
			var val any

			// If Decode reads beyond what w.Write writes above,
			// it will block, and the test will deadlock.
			if err := NewDecoder(r).Decode(&val); err != nil {
				t.Errorf("%s: NewDecoder(%s).Decode error: %v", tt.Where, tt.in, err)
			}
			r.Close()
			w.Close()
		})
	}
}

type decodeThis struct {
	v any
}

func TestDecodeInStream(t *testing.T) {
	tests := []struct {
		CaseName
		json      string
		expTokens []any
	}{
		// streaming token cases
		{CaseName: Name(""), json: `10`, expTokens: []any{float64(10)}},
		{CaseName: Name(""), json: ` [10] `, expTokens: []any{
			Delim('['), float64(10), Delim(']')}},
		{CaseName: Name(""), json: ` [false,10,"b"] `, expTokens: []any{
			Delim('['), false, float64(10), "b", Delim(']')}},
		{CaseName: Name(""), json: `{ "a": 1 }`, expTokens: []any{
			Delim('{'), "a", float64(1), Delim('}')}},
		{CaseName: Name(""), json: `{"a": 1, "b":"3"}`, expTokens: []any{
			Delim('{'), "a", float64(1), "b", "3", Delim('}')}},
		{CaseName: Name(""), json: ` [{"a": 1},{"a": 2}] `, expTokens: []any{
			Delim('['),
			Delim('{'), "a", float64(1), Delim('}'),
			Delim('{'), "a", float64(2), Delim('}'),
			Delim(']')}},
		{CaseName: Name(""), json: `{"obj": {"a": 1}}`, expTokens: []any{
			Delim('{'), "obj", Delim('{'), "a", float64(1), Delim('}'),
			Delim('}')}},
		{CaseName: Name(""), json: `{"obj": [{"a": 1}]}`, expTokens: []any{
			Delim('{'), "obj", Delim('['),
			Delim('{'), "a", float64(1), Delim('}'),
			Delim(']'), Delim('}')}},

		// streaming tokens with intermittent Decode()
		{CaseName: Name(""), json: `{ "a": 1 }`, expTokens: []any{
			Delim('{'), "a",
			decodeThis{float64(1)},
			Delim('}')}},
		{CaseName: Name(""), json: ` [ { "a" : 1 } ] `, expTokens: []any{
			Delim('['),
			decodeThis{map[string]any{"a": float64(1)}},
			Delim(']')}},
		{CaseName: Name(""), json: ` [{"a": 1},{"a": 2}] `, expTokens: []any{
			Delim('['),
			decodeThis{map[string]any{"a": float64(1)}},
			decodeThis{map[string]any{"a": float64(2)}},
			Delim(']')}},
		{CaseName: Name(""), json: `{ "obj" : [ { "a" : 1 } ] }`, expTokens: []any{
			Delim('{'), "obj", Delim('['),
			decodeThis{map[string]any{"a": float64(1)}},
			Delim(']'), Delim('}')}},

		{CaseName: Name(""), json: `{"obj": {"a": 1}}`, expTokens: []any{
			Delim('{'), "obj",
			decodeThis{map[string]any{"a": float64(1)}},
			Delim('}')}},
		{CaseName: Name(""), json: `{"obj": [{"a": 1}]}`, expTokens: []any{
			Delim('{'), "obj",
			decodeThis{[]any{
				map[string]any{"a": float64(1)},
			}},
			Delim('}')}},
		{CaseName: Name(""), json: ` [{"a": 1} {"a": 2}] `, expTokens: []any{
			Delim('['),
			decodeThis{map[string]any{"a": float64(1)}},
			decodeThis{&SyntaxError{"expected comma after array element", 11, "", 0}},
		}},
		{CaseName: Name(""), json: `{ "` + strings.Repeat("a", 513) + `" 1 }`, expTokens: []any{
			Delim('{'), strings.Repeat("a", 513),
			decodeThis{&SyntaxError{"expected colon after object key", 518, "", 0}},
		}},
		{CaseName: Name(""), json: `{ "\a" }`, expTokens: []any{
			Delim('{'),
			&SyntaxError{invalidChar: 'a', invalidCharContext: "in string escape code", Offset: 3},
		}},
		{CaseName: Name(""), json: ` \a`, expTokens: []any{
			&SyntaxError{invalidChar: '\\', invalidCharContext: "looking for beginning of value", Offset: 1},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			dec := NewDecoder(strings.NewReader(tt.json))
			for i, want := range tt.expTokens {
				var got any
				var err error

				if dt, ok := want.(decodeThis); ok {
					want = dt.v
					err = dec.Decode(&got)
				} else {
					got, err = dec.Token()
				}
				if errWant, ok := want.(error); ok {
					if err == nil || !reflect.DeepEqual(err, errWant) {
						t.Fatalf("%s:\n\tinput: %s\n\tgot error:  %v\n\twant error: %v", tt.Where, tt.json, err, errWant)
					}
					break
				} else if err != nil {
					t.Fatalf("%s:\n\tinput: %s\n\tgot error:  %v\n\twant error: nil", tt.Where, tt.json, err)
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("%s: token %d:\n\tinput: %s\n\tgot:  %T(%v)\n\twant: %T(%v)", tt.Where, i, tt.json, got, got, want, want)
				}
			}
		})
	}
}

// Test from golang.org/issue/11893
func TestHTTPDecoding(t *testing.T) {
	const raw = `{ "foo": "bar" }`

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(raw))
	}))
	defer ts.Close()
	res, err := http.Get(ts.URL)
	if err != nil {
		log.Fatalf("http.Get error: %v", err)
	}
	defer res.Body.Close()

	foo := struct {
		Foo string
	}{}

	d := NewDecoder(res.Body)
	err = d.Decode(&foo)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if foo.Foo != "bar" {
		t.Errorf(`Decode: got %q, want "bar"`, foo.Foo)
	}

	// make sure we get the EOF the second time
	err = d.Decode(&foo)
	if err != io.EOF {
		t.Errorf("Decode error:\n\tgot:  %v\n\twant: io.EOF", err)
	}
}
