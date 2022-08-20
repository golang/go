// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"runtime/debug"
	"strings"
	"testing"
)

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
		var buf bytes.Buffer
		enc := NewEncoder(&buf)
		// Check that enc.SetIndent("", "") turns off indentation.
		enc.SetIndent(">", ".")
		enc.SetIndent("", "")
		for j, v := range streamTest[0:i] {
			if err := enc.Encode(v); err != nil {
				t.Fatalf("encode #%d: %v", j, err)
			}
		}
		if have, want := buf.String(), nlines(streamEncoded, i); have != want {
			t.Errorf("encoding %d items: mismatch", i)
			diff(t, []byte(have), []byte(want))
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
		t.Errorf("Encode(dummy) == nil; want error")
	}

	type Data struct {
		A string
		I int
	}
	data := Data{A: "a", I: 1}
	if err := enc.Encode(data); err != nil {
		t.Errorf("Marshal(%v) = %v", data, err)
	}

	var data2 Data
	if err := Unmarshal(buf.Bytes(), &data2); err != nil {
		t.Errorf("Unmarshal(%v) = %v", data2, err)
	}
	if data2 != data {
		t.Errorf("expect: %v, but get: %v", data, data2)
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
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	enc.SetIndent(">", ".")
	for _, v := range streamTest {
		enc.Encode(v)
	}
	if have, want := buf.String(), streamEncodedIndent; have != want {
		t.Error("indented encoding mismatch")
		diff(t, []byte(have), []byte(want))
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

	for _, tt := range []struct {
		name       string
		v          any
		wantEscape string
		want       string
	}{
		{"c", c, `"\u003c\u0026\u003e"`, `"<&>"`},
		{"ct", ct, `"\"\u003c\u0026\u003e\""`, `"\"<&>\""`},
		{`"<&>"`, "<&>", `"\u003c\u0026\u003e"`, `"<&>"`},
		{
			"tagStruct", tagStruct,
			`{"\u003c\u003e\u0026#! ":0,"Invalid":0}`,
			`{"<>&#! ":0,"Invalid":0}`,
		},
		{
			`"<str>"`, marshalerStruct,
			`{"NonPtr":"\u003cstr\u003e","Ptr":"\u003cstr\u003e"}`,
			`{"NonPtr":"<str>","Ptr":"<str>"}`,
		},
		{
			"stringOption", stringOption,
			`{"bar":"\"\\u003chtml\\u003efoobar\\u003c/html\\u003e\""}`,
			`{"bar":"\"<html>foobar</html>\""}`,
		},
	} {
		var buf bytes.Buffer
		enc := NewEncoder(&buf)
		if err := enc.Encode(tt.v); err != nil {
			t.Errorf("Encode(%s): %s", tt.name, err)
			continue
		}
		if got := strings.TrimSpace(buf.String()); got != tt.wantEscape {
			t.Errorf("Encode(%s) = %#q, want %#q", tt.name, got, tt.wantEscape)
		}
		buf.Reset()
		enc.SetEscapeHTML(false)
		if err := enc.Encode(tt.v); err != nil {
			t.Errorf("SetEscapeHTML(false) Encode(%s): %s", tt.name, err)
			continue
		}
		if got := strings.TrimSpace(buf.String()); got != tt.want {
			t.Errorf("SetEscapeHTML(false) Encode(%s) = %#q, want %#q",
				tt.name, got, tt.want)
		}
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
				t.Fatalf("decode #%d/%d: %v", j, i, err)
			}
		}
		if !reflect.DeepEqual(out, streamTest[0:i]) {
			t.Errorf("decoding %d items: mismatch", i)
			for j := range out {
				if !reflect.DeepEqual(out[j], streamTest[j]) {
					t.Errorf("#%d: have %v want %v", j, out[j], streamTest[j])
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
		t.Errorf("Name = %q; want Gopher", m.Name)
	}
	rest, err := io.ReadAll(d.Buffered())
	if err != nil {
		t.Fatal(err)
	}
	if g, w := string(rest), " extra "; g != w {
		t.Errorf("Remaining = %q; want %q", g, w)
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
	const msg = `{"X":0.1,"Id":["\u0056",null],"Y":0.2}`
	err := Unmarshal([]byte(msg), &data)
	if err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if string([]byte(data.Id)) != raw {
		t.Fatalf("Raw mismatch: have %#q want %#q", []byte(data.Id), raw)
	}
	b, err := Marshal(&data)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(b) != msg {
		t.Fatalf("Marshal: have %#q want %#q", b, msg)
	}
}

func TestNullRawMessage(t *testing.T) {
	var data struct {
		X     float64
		Id    RawMessage
		IdPtr *RawMessage
		Y     float32
	}
	const msg = `{"X":0.1,"Id":null,"IdPtr":null,"Y":0.2}`
	err := Unmarshal([]byte(msg), &data)
	if err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if want, got := "null", string(data.Id); want != got {
		t.Fatalf("Raw mismatch: have %q, want %q", got, want)
	}
	if data.IdPtr != nil {
		t.Fatalf("Raw pointer mismatch: have non-nil, want nil")
	}
	b, err := Marshal(&data)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(b) != msg {
		t.Fatalf("Marshal: have %#q want %#q", b, msg)
	}
}

var blockingTests = []string{
	`{"x": 1}`,
	`[1, 2, 3]`,
}

func TestBlocking(t *testing.T) {
	for _, enc := range blockingTests {
		r, w := net.Pipe()
		go w.Write([]byte(enc))
		var val any

		// If Decode reads beyond what w.Write writes above,
		// it will block, and the test will deadlock.
		if err := NewDecoder(r).Decode(&val); err != nil {
			t.Errorf("decoding %s: %v", enc, err)
		}
		r.Close()
		w.Close()
	}
}

type tokenStreamCase struct {
	json      string
	expTokens []any
}

type decodeThis struct {
	v any
}

var tokenStreamCases = []tokenStreamCase{
	// streaming token cases
	{json: `10`, expTokens: []any{float64(10)}},
	{json: ` [10] `, expTokens: []any{
		Delim('['), float64(10), Delim(']')}},
	{json: ` [false,10,"b"] `, expTokens: []any{
		Delim('['), false, float64(10), "b", Delim(']')}},
	{json: `{ "a": 1 }`, expTokens: []any{
		Delim('{'), "a", float64(1), Delim('}')}},
	{json: `{"a": 1, "b":"3"}`, expTokens: []any{
		Delim('{'), "a", float64(1), "b", "3", Delim('}')}},
	{json: ` [{"a": 1},{"a": 2}] `, expTokens: []any{
		Delim('['),
		Delim('{'), "a", float64(1), Delim('}'),
		Delim('{'), "a", float64(2), Delim('}'),
		Delim(']')}},
	{json: `{"obj": {"a": 1}}`, expTokens: []any{
		Delim('{'), "obj", Delim('{'), "a", float64(1), Delim('}'),
		Delim('}')}},
	{json: `{"obj": [{"a": 1}]}`, expTokens: []any{
		Delim('{'), "obj", Delim('['),
		Delim('{'), "a", float64(1), Delim('}'),
		Delim(']'), Delim('}')}},

	// streaming tokens with intermittent Decode()
	{json: `{ "a": 1 }`, expTokens: []any{
		Delim('{'), "a",
		decodeThis{float64(1)},
		Delim('}')}},
	{json: ` [ { "a" : 1 } ] `, expTokens: []any{
		Delim('['),
		decodeThis{map[string]any{"a": float64(1)}},
		Delim(']')}},
	{json: ` [{"a": 1},{"a": 2}] `, expTokens: []any{
		Delim('['),
		decodeThis{map[string]any{"a": float64(1)}},
		decodeThis{map[string]any{"a": float64(2)}},
		Delim(']')}},
	{json: `{ "obj" : [ { "a" : 1 } ] }`, expTokens: []any{
		Delim('{'), "obj", Delim('['),
		decodeThis{map[string]any{"a": float64(1)}},
		Delim(']'), Delim('}')}},

	{json: `{"obj": {"a": 1}}`, expTokens: []any{
		Delim('{'), "obj",
		decodeThis{map[string]any{"a": float64(1)}},
		Delim('}')}},
	{json: `{"obj": [{"a": 1}]}`, expTokens: []any{
		Delim('{'), "obj",
		decodeThis{[]any{
			map[string]any{"a": float64(1)},
		}},
		Delim('}')}},
	{json: ` [{"a": 1} {"a": 2}] `, expTokens: []any{
		Delim('['),
		decodeThis{map[string]any{"a": float64(1)}},
		decodeThis{&SyntaxError{"expected comma after array element", 11}},
	}},
	{json: `{ "` + strings.Repeat("a", 513) + `" 1 }`, expTokens: []any{
		Delim('{'), strings.Repeat("a", 513),
		decodeThis{&SyntaxError{"expected colon after object key", 518}},
	}},
	{json: `{ "\a" }`, expTokens: []any{
		Delim('{'),
		&SyntaxError{"invalid character 'a' in string escape code", 3},
	}},
	{json: ` \a`, expTokens: []any{
		&SyntaxError{"invalid character '\\\\' looking for beginning of value", 1},
	}},
}

func TestDecodeInStream(t *testing.T) {
	for ci, tcase := range tokenStreamCases {

		dec := NewDecoder(strings.NewReader(tcase.json))
		for i, etk := range tcase.expTokens {

			var tk any
			var err error

			if dt, ok := etk.(decodeThis); ok {
				etk = dt.v
				err = dec.Decode(&tk)
			} else {
				tk, err = dec.Token()
			}
			if experr, ok := etk.(error); ok {
				if err == nil || !reflect.DeepEqual(err, experr) {
					t.Errorf("case %v: Expected error %#v in %q, but was %#v", ci, experr, tcase.json, err)
				}
				break
			} else if err == io.EOF {
				t.Errorf("case %v: Unexpected EOF in %q", ci, tcase.json)
				break
			} else if err != nil {
				t.Errorf("case %v: Unexpected error '%#v' in %q", ci, err, tcase.json)
				break
			}
			if !reflect.DeepEqual(tk, etk) {
				t.Errorf(`case %v: %q @ %v expected %T(%v) was %T(%v)`, ci, tcase.json, i, etk, etk, tk, tk)
				break
			}
		}
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
		log.Fatalf("GET failed: %v", err)
	}
	defer res.Body.Close()

	foo := struct {
		Foo string
	}{}

	d := NewDecoder(res.Body)
	err = d.Decode(&foo)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if foo.Foo != "bar" {
		t.Errorf("decoded %q; want \"bar\"", foo.Foo)
	}

	// make sure we get the EOF the second time
	err = d.Decode(&foo)
	if err != io.EOF {
		t.Errorf("err = %v; want io.EOF", err)
	}
}
