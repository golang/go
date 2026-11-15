// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	stdjson "encoding/json"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"
	"unsafe"
)

type user struct {
	ID   int
	Name string
}

func collectKeys(data string) ([]string, error) {
	var keys []string
	s := NewScanner([]byte(data), (*user)(nil))
	if !s.Object() {
		return nil, s.End()
	}
	for s.Next() {
		keys = append(keys, s.Key())
		s.Skip()
	}
	return keys, s.Err()
}

func TestObjectIteration(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{`{}`, nil},
		{` { } `, nil},
		{`{"a":1}`, []string{"a"}},
		{`{"a":1,"b":[1,2],"c":{"d":null},"e":"x"}`, []string{"a", "b", "c", "e"}},
		{`{"a":1,"a":2,"A":3}`, []string{"a", "a", "A"}},
		{`{"a\u0062":1,"\n":2,"\"":3,"\\":4,"\/":5,"\ud83d\ude00":6}`, []string{"ab", "\n", "\"", "\\", "/", "😀"}},
		{`{"":1}`, []string{""}},
		{"{\"a\xffb\":1}", []string{"a\ufffdb"}},
		{"{\"\xc3\":1,\"\xe2\x82\":2}", []string{"\ufffd", "\ufffd\ufffd"}},
		{`{"\ud800":1,"\udc00x":2,"\ud800\u0041":3}`, []string{"\ufffd", "\ufffdx", "\ufffdA"}},
		{`{ "a" : 1 , "b" : 2 }`, []string{"a", "b"}},
		{"\n\t{\r\n\"k\"\t:\n1\r,\"l\":2}\n", []string{"k", "l"}},
	}
	for _, c := range cases {
		got, err := collectKeys(c.in)
		if err != nil {
			t.Errorf("%q: %v", c.in, err)
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("%q: keys = %q, want %q", c.in, got, c.want)
		}
		var viaStd map[string]stdjson.RawMessage
		if err := stdjson.Unmarshal([]byte(c.in), &viaStd); err != nil {
			t.Errorf("%q: encoding/json: %v", c.in, err)
			continue
		}
		for _, k := range got {
			if _, ok := viaStd[k]; !ok {
				t.Errorf("%q: key %q unknown to encoding/json", c.in, k)
			}
		}
	}
}

func aliases(data []byte, k string) bool {
	if k == "" {
		return false
	}
	p := uintptr(unsafe.Pointer(unsafe.StringData(k)))
	lo := uintptr(unsafe.Pointer(&data[0]))
	return p >= lo && p < lo+uintptr(len(data))
}

func TestKeyIsZeroCopyWithoutEscapes(t *testing.T) {
	data := []byte(`{"plain":1,"esc\u0061ped":2,"ok":3}`)
	s := NewScanner(data, (*user)(nil))
	s.Object()
	for s.Next() {
		k := s.Key()
		inData := aliases(data, k)
		switch k {
		case "plain", "ok":
			if !inData {
				t.Errorf("key %q was copied", k)
			}
		case "escaped":
			if inData {
				t.Errorf("escaped key %q aliases the input", k)
			}
		default:
			t.Errorf("unexpected key %q", k)
		}
		s.Skip()
	}
	if err := s.End(); err != nil {
		t.Fatal(err)
	}
}

func TestKeyPlainKeyAllocFree(t *testing.T) {
	data := []byte(`{"id":1,"name":"n","extra":[1,2]}`)
	allocs := testing.AllocsPerRun(100, func() {
		s := NewScanner(data, (*user)(nil))
		if !s.Object() {
			t.Fatal("Object")
		}
		for s.Next() {
			switch s.Key() {
			case "id", "name":
				s.Skip()
			default:
				s.Skip()
			}
		}
		if s.End() != nil {
			t.Fatal("End")
		}
	})
	if allocs != 0 {
		t.Fatalf("iteration allocated %v times", allocs)
	}
}

func foldOf(t *testing.T, key string) string {
	t.Helper()
	b, err := stdjson.Marshal(key)
	if err != nil {
		t.Fatal(err)
	}
	s := NewScanner([]byte("{"+string(b)+":0}"), (*user)(nil))
	if !s.Object() || !s.Next() {
		t.Fatalf("no member for key %q", key)
	}
	f := s.FoldKey()
	return strings.Clone(f)
}

func TestFoldKeyMatchesEqualFold(t *testing.T) {
	corpus := []string{
		"", "a", "A", "id", "ID", "Id", "iD", "userName", "USERNAME", "username", "user_name", "k", "K", "\u212a", "s", "S", "\u017f",
		"\u03c3", "\u03c2", "\u03a3", "\u01c5", "\u01c4", "\u01c6", "\u00e9", "\u00c9", "\u1e9e", "\u00df", "ss", "SS", "\u2126", "\u03c9", "\u03a9",
		"straße", "STRASSE", "日本", "日本語", "Ǆ", "ǆ", "\u0130", "i", "I", "\u0131", "ǅungla", "ǄUNGLA", "k\u212a", "Kk", "kK", "ſſ", "SS", "ſs",
		"a-b", "A-B", "a_b", "x1", "X1", "\u00e9\u00c9", "\u0345", "\u03b9", "\u1fbe", "\u0399",
	}
	for _, a := range corpus {
		for _, b := range corpus {
			got := foldOf(t, a) == foldOf(t, b)
			want := bytes.EqualFold([]byte(a), []byte(b))
			if got != want {
				t.Errorf("fold(%q)==fold(%q) is %v, bytes.EqualFold says %v", a, b, got, want)
			}
		}
	}
}

func TestFoldKeyValues(t *testing.T) {
	cases := map[string]string{"id": "ID", "Name": "NAME", "a-b": "A-B", "\u212a": "K", "\u017f": "S", "é": "É", "": ""}
	for in, want := range cases {
		if got := foldOf(t, in); got != want {
			t.Errorf("FoldKey(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestFoldKeyAllocFreeUpTo32Bytes(t *testing.T) {
	short := []byte(`{"someFairlyLongFieldName_01234":1}`)
	allocs := testing.AllocsPerRun(100, func() {
		s := NewScanner(short, (*user)(nil))
		s.Object()
		s.Next()
		switch s.FoldKey() {
		case "SOMEFAIRLYLONGFIELDNAME_01234":
		default:
			s.Skip()
		}
	})
	if allocs != 0 {
		t.Fatalf("FoldKey of a 29 byte key allocated %v times", allocs)
	}
	long := []byte(`{"` + strings.Repeat("longKey", 10) + `":1}`)
	s := NewScanner(long, (*user)(nil))
	s.Object()
	s.Next()
	if got, want := s.FoldKey(), strings.ToUpper(strings.Repeat("longKey", 10)); got != want {
		t.Errorf("FoldKey of a long key = %q, want %q", got, want)
	}
}

func TestNullSkipAndEnd(t *testing.T) {
	s := NewScanner([]byte(`{"a":null,"b":nullx}`), (*user)(nil))
	s.Object()
	if !s.Next() || s.Key() != "a" || !s.Null() {
		t.Fatal("first member should be null")
	}
	if s.Next() {
		if s.Key() != "b" {
			t.Fatal("second key")
		}
		if s.Null() {
			t.Fatal("nullx must not be taken as null")
		}
		s.Skip()
		if s.Next() {
			t.Fatal("Next after garbage")
		}
	}
	if s.Err() == nil {
		t.Fatal("expected a syntax error for nullx")
	}
	var raw stdjson.RawMessage
	if want := stdjson.Unmarshal([]byte(`{"a":null,"b":nullx}`), &raw); !sameError(s.Err(), want) {
		t.Fatalf("Err() = %v, want %v", s.Err(), want)
	}
	if s.End() != s.Err() {
		t.Fatal("End() must return Err() first")
	}
}

func TestNullAtEndOfInput(t *testing.T) {
	s := NewScanner([]byte(`{"a":null`), (*user)(nil))
	s.Object()
	s.Next()
	if !s.Null() {
		t.Fatal("null at EOF is still null")
	}
	if s.Next() {
		t.Fatal("Next at EOF")
	}
	if s.Err() == nil || s.Err().Error() != "unexpected end of JSON input" {
		t.Fatalf("Err() = %v", s.Err())
	}
}

func TestObjectOnOtherKinds(t *testing.T) {
	inputs := []string{
		`null`, ` null `, `[]`, `[1,2]`, ` [ ] `, `"s"`, `""`, `1`, `-1.5`, `1e5`, `true`, `false`, `[{"a":1}]`,
		`{`, `{}x`, `[1,`, `"abc`, `tru`, ``, ` `, `nul`, `nullx`, `01`, `[1] 2`, `1 2`, `{"a":1} {"b":2}`, `{"a":1}x`,
	}
	for _, in := range inputs {
		s := NewScanner([]byte(in), (*user)(nil))
		isObj := s.Object()
		if isObj {
			for s.Next() {
				s.Skip()
			}
		}
		gotErr := s.End()
		var want user
		wantErr := stdjson.Unmarshal([]byte(in), &want)
		if in == "null" || in == " null " {
			if gotErr != nil || wantErr != nil || isObj {
				t.Errorf("%q: Object=%v End=%v std=%v", in, isObj, gotErr, wantErr)
			}
			continue
		}
		if !sameError(gotErr, wantErr) {
			t.Errorf("%q: End() = %#v (%v), std = %#v (%v)", in, gotErr, gotErr, wantErr, wantErr)
		}
	}
}

func TestObjectTypeErrorFields(t *testing.T) {
	cases := []struct {
		in    string
		value string
		off   int64
	}{
		{`[1]`, "array", 1}, {`"s"`, "string", 3}, {`12`, "number", 2}, {`true`, "bool", 4}, {`false`, "bool", 5}, {` [] `, "array", 2},
	}
	for _, c := range cases {
		s := NewScanner([]byte(c.in), (*user)(nil))
		if s.Object() {
			t.Fatalf("%q: Object() = true", c.in)
		}
		ute, ok := s.End().(*stdjson.UnmarshalTypeError)
		if !ok {
			t.Fatalf("%q: End() = %#v", c.in, s.End())
		}
		if ute.Value != c.value || ute.Type != reflect.TypeFor[user]() || ute.Offset != c.off || ute.Struct != "" || ute.Field != "" {
			t.Errorf("%q: %#v", c.in, ute)
		}
	}
}

func TestErrorsAreSticky(t *testing.T) {
	s := NewScanner([]byte(`{"a":1,]`), (*user)(nil))
	s.Object()
	s.Next()
	s.Skip()
	if s.Next() {
		t.Fatal("Next after error")
	}
	first := s.Err()
	if first == nil {
		t.Fatal("no error")
	}
	s.Skip()
	var n int
	Int(&s, &n)
	if s.Next() || s.Err() != first {
		t.Fatal("error not sticky")
	}
}

func TestNextAfterCloseStaysFalse(t *testing.T) {
	s := NewScanner([]byte(`{"a":1}`), (*user)(nil))
	s.Object()
	for s.Next() {
		s.Skip()
	}
	if s.Next() || s.Err() != nil || s.Saved() != nil {
		t.Fatal("Next after close")
	}
}

func TestTrailingDataAfterObject(t *testing.T) {
	for _, in := range []string{`{} x`, `{"a":1} 2`, `{"a":1}{}`, `{"a":1},`} {
		keys, err := collectKeys(in)
		var raw stdjson.RawMessage
		want := stdjson.Unmarshal([]byte(in), &raw)
		if !sameError(err, want) {
			t.Errorf("%q: keys %q err %v, want %v", in, keys, err, want)
		}
	}
}

func TestSkipValidates(t *testing.T) {
	for _, in := range []string{`{"a":[1,2,{"b":tru}]}`, `{"a":{"b":}}`, `{"a":"\q"}`, `{"a":[1 2]}`, `{"a":{"b" 1}}`, `{"a":-}`, `{"a":1.e2}`, `{"a":.5}`, `{"a":+1}`, `{"a":'x'}`, `{"a":NaN}`, `{"a":{,}}`} {
		_, err := collectKeys(in)
		var raw stdjson.RawMessage
		want := stdjson.Unmarshal([]byte(in), &raw)
		if want == nil {
			t.Fatalf("%q valid", in)
		}
		if !sameError(err, want) {
			t.Errorf("%q: err %v, want %v", in, err, want)
		}
	}
}

func TestSkipAcceptsEveryValueKind(t *testing.T) {
	in := `{"a":1,"b":-0.5e+3,"c":"s\u00e9\n","d":true,"e":false,"f":null,"g":[],"h":{},"i":[1,[2,{"x":[]}],"y"],"j":{"k":{"l":[null]}}}`
	keys, err := collectKeys(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) != 10 {
		t.Fatalf("keys = %q", keys)
	}
}

func syntaxCorpus() []string {
	valid := `{"id":12,"name":"a\u00e9\n","tags":["x","y",[1,2.5e-3,-0]],"ok":true,"nil":null,"obj":{"k":{"z":[]}},"s":"q"}`
	var c []string
	for i := range len(valid) + 1 {
		c = append(c, valid[:i])
	}
	c = append(c, valid+" ", valid+"x", valid+",", valid+"}", " \t\r\n"+valid+" \t\r\n", valid+valid)
	for i := range len(valid) {
		c = append(c, valid[:i]+valid[i+1:])
		for _, b := range []string{"\x00", "\x7f", "\xff", "'", "}", "]", ",", ":", "\"", "\\", "0", "-", "e", "t", " "} {
			c = append(c, valid[:i]+b+valid[i+1:])
			c = append(c, valid[:i]+b+valid[i:])
		}
	}
	c = append(c,
		`{"a":01}`, `{"a":1.}`, `{"a":-}`, `{"a":1e}`, `{"a":1e+}`, `{"a":"\x"}`, "{\"a\":\"\x01\"}", "{\"a\":\"\n\"}", `{"a":"\u12"}`, `{"a":"\u12G4"}`,
		`{"a":1,}`, `{,}`, `{"a"}`, `{"a":}`, `{"a" 1}`, `{a:1}`, `{'a':1}`, `{"a":1 "b":2}`, `{"a":[1,]}`, `{"a":[,1]}`, `{"a":[1 2]}`, `{"a":{"b":1]}`,
		`{"a":tru}`, `{"a":truee}`, `{"a":nulll}`, `{"a":fals}`, `{"a":True}`, `{"a":NaN}`, `{"a":Infinity}`, `{"a":0x10}`, `{"a":1_0}`,
		`{"a":"\ud800"}`, `{"a":"\udc00\ud800"}`, "{\"a\":\"\xff\"}", "{\"\xff\":1}",
		"\ufeff{}", "{}\x00", "\x00", "{\"a\":\x00}",
		`{"a":`+strings.Repeat("[", 9999)+strings.Repeat("]", 9999)+`}`,
		`{"a":`+strings.Repeat("[", 10000)+strings.Repeat("]", 10000)+`}`,
		`{"a":`+strings.Repeat("[", 10001),
		`{"a":`+strings.Repeat(`{"b":`, 9999)+`1`+strings.Repeat("}", 9999)+`}`,
		`{"a":`+strings.Repeat(`{"b":`, 10000)+`1`+strings.Repeat("}", 10000)+`}`,
		`{"a":`+strings.Repeat(`[{"b":`, 5000)+`1`+strings.Repeat("}]", 5000)+`}`,
	)
	return c
}

func checkSyntaxParity(t testing.TB, in string) {
	t.Helper()
	var raw stdjson.RawMessage
	want := stdjson.Unmarshal([]byte(in), &raw)
	s := NewScanner([]byte(in), (*user)(nil))
	if s.Object() {
		for s.Next() {
			s.Skip()
		}
	}
	got := s.Err()
	if !sameError(got, want) {
		t.Fatalf("%q: Err() = %#v (%v), encoding/json = %#v (%v)", clip(in), got, got, want, want)
	}
}

func clip(s string) string {
	if len(s) > 80 {
		return s[:80] + "..."
	}
	return s
}

func TestSyntaxErrorParityCorpus(t *testing.T) {
	for _, in := range syntaxCorpus() {
		checkSyntaxParity(t, in)
	}
}

func FuzzScannerSyntax(f *testing.F) {
	for _, in := range syntaxCorpus() {
		if len(in) < 400 {
			f.Add(in)
		}
	}
	f.Fuzz(func(t *testing.T, in string) {
		checkSyntaxParity(t, in)
	})
}

type fallbackTarget struct {
	M map[string]int
	L []int
	P *int
	I any
}

func TestFallback(t *testing.T) {
	var ft fallbackTarget
	data := `{"m":{"a":1,"b":2},"l":[1,2,3],"p":5,"i":{"x":[true,null]},"u":1}`
	s := NewScanner([]byte(data), (*fallbackTarget)(nil))
	s.Object()
	for s.Next() {
		switch s.Key() {
		case "m":
			s.Fallback(&ft.M)
		case "l":
			s.Fallback(&ft.L)
		case "p":
			s.Fallback(&ft.P)
		case "i":
			s.Fallback(&ft.I)
		default:
			s.Skip()
		}
	}
	if err := s.End(); err != nil {
		t.Fatal(err)
	}
	var want fallbackTarget
	if err := stdjson.Unmarshal([]byte(data), &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ft, want) {
		t.Fatalf("got %#v, want %#v", ft, want)
	}
}

func TestFallbackSavesTypeErrorAndContinues(t *testing.T) {
	var ft fallbackTarget
	data := `{"L":[1,"x",3],"M":{"k":"v"},"P":7}`
	s := NewScanner([]byte(data), (*fallbackTarget)(nil))
	s.Object()
	for s.Next() {
		switch s.Key() {
		case "M":
			s.Fallback(&ft.M)
		case "L":
			s.Fallback(&ft.L)
		case "P":
			s.Fallback(&ft.P)
		}
	}
	if s.Err() != nil {
		t.Fatalf("Err() = %v", s.Err())
	}
	if ft.P == nil || *ft.P != 7 {
		t.Fatal("decoding did not continue after the saved error")
	}
	ute, ok := s.Saved().(*stdjson.UnmarshalTypeError)
	if !ok {
		t.Fatalf("Saved() = %#v", s.Saved())
	}
	wantField := "L"
	if stdIsV2 {
		wantField = "L.1"
	}
	if ute.Struct != "fallbackTarget" || ute.Field != wantField || ute.Value != "string" || ute.Type != reflect.TypeFor[int]() || ute.Offset != 11 {
		t.Fatalf("saved error = %#v", ute)
	}
	var want fallbackTarget
	wantErr := stdjson.Unmarshal([]byte(data), &want)
	if !sameError(s.Saved(), wantErr) {
		t.Fatalf("Saved() = %v, encoding/json = %v", s.Saved(), wantErr)
	}
	if !reflect.DeepEqual(ft.L, want.L) {
		t.Fatalf("L = %v, want %v", ft.L, want.L)
	}
}

func TestFallbackFirstSavedErrorWins(t *testing.T) {
	var a, b []int
	s := NewScanner([]byte(`{"a":["x"],"b":[true]}`), (*fallbackTarget)(nil))
	s.Object()
	for s.Next() {
		switch s.Key() {
		case "a":
			s.Fallback(&a)
		case "b":
			s.Fallback(&b)
		}
	}
	wantField := "a"
	if stdIsV2 {
		wantField = "a.0"
	}
	ute, ok := s.Saved().(*stdjson.UnmarshalTypeError)
	if !ok || ute.Field != wantField || ute.Value != "string" {
		t.Fatalf("Saved() = %#v", s.Saved())
	}
}

func TestFallbackNonTypeErrorIsFatal(t *testing.T) {
	var notPointer int
	s := NewScanner([]byte(`{"a":1,"b":2}`), (*fallbackTarget)(nil))
	s.Object()
	for s.Next() {
		s.Fallback(notPointer)
	}
	if s.Err() == nil {
		t.Fatal("expected a fatal error for a non-pointer destination")
	}
	if s.Saved() != nil {
		t.Fatalf("Saved() = %v", s.Saved())
	}
	if _, ok := s.Err().(*stdjson.InvalidUnmarshalError); !ok {
		t.Fatalf("Err() = %#v", s.Err())
	}
}

func TestFallbackNestedStructFieldPath(t *testing.T) {
	type inner struct{ X int }
	type outer struct{ In inner }
	var o outer
	data := `{"In":{"X":"bad"}}`
	s := NewScanner([]byte(data), (*outer)(nil))
	s.Object()
	for s.Next() {
		s.Fallback(&o.In)
	}
	ute, ok := s.Saved().(*stdjson.UnmarshalTypeError)
	if !ok {
		t.Fatalf("Saved() = %#v", s.Saved())
	}
	if ute.Field != "In.X" || ute.Value != "string" || ute.Offset != 16 {
		t.Fatalf("saved error = %#v", ute)
	}
}

func TestFallbackSyntaxError(t *testing.T) {
	var l []int
	data := `{"l":[1,2,}`
	s := NewScanner([]byte(data), (*fallbackTarget)(nil))
	s.Object()
	for s.Next() {
		s.Fallback(&l)
	}
	var raw stdjson.RawMessage
	if want := stdjson.Unmarshal([]byte(data), &raw); !sameError(s.Err(), want) {
		t.Fatalf("Err() = %v, want %v", s.Err(), want)
	}
}

type fallbackQRecorder struct {
	got string
}

func (r *fallbackQRecorder) UnmarshalJSON(b []byte) error {
	r.got = string(b)
	return nil
}

func TestFallbackQUnwrapsStringOption(t *testing.T) {
	var r fallbackQRecorder
	s := NewScanner([]byte(`{"v":"7"}`), (*fallbackTarget)(nil))
	if !s.Object() {
		t.Fatalf("Object failed: %v", s.Err())
	}
	for s.Next() {
		FallbackQ(&s, &r)
	}
	if err := s.End(); err != nil {
		t.Fatalf("End() = %v, Saved() = %v", err, s.Saved())
	}
	if r.got != "7" {
		t.Fatalf("UnmarshalJSON received %q, want %q (inner content, not outer token)", r.got, "7")
	}
}

func TestFallbackQNullLeavesValue(t *testing.T) {
	var n int = 5
	s := NewScanner([]byte(`{"v":null}`), (*fallbackTarget)(nil))
	if !s.Object() {
		t.Fatalf("Object failed: %v", s.Err())
	}
	for s.Next() {
		FallbackQ(&s, &n)
	}
	if err := s.End(); err != nil {
		t.Fatalf("End() = %v, Saved() = %v", err, s.Saved())
	}
	if n != 5 {
		t.Fatalf("null changed value to %d, want unchanged 5", n)
	}
}

func TestFallbackQBareValueSavesError(t *testing.T) {
	var r fallbackQRecorder
	s := NewScanner([]byte(`{"v":7}`), (*fallbackTarget)(nil))
	if !s.Object() {
		t.Fatalf("Object failed: %v", s.Err())
	}
	for s.Next() {
		FallbackQ(&s, &r)
	}
	if s.Err() != nil {
		t.Fatalf("Err() = %v, want nil (bare should be saved, not fatal)", s.Err())
	}
	want := "json: invalid use of ,string struct tag, trying to unmarshal unquoted value into json.fallbackQRecorder"
	if s.Saved() == nil || s.Saved().Error() != want {
		t.Fatalf("Saved() = %v, want %q", s.Saved(), want)
	}
}

func TestSavedAndEndOrdering(t *testing.T) {
	s := NewScanner([]byte(`{"a":"x"}`), (*user)(nil))
	s.Object()
	var n int
	for s.Next() {
		Int(&s, &n)
	}
	if s.Err() != nil || s.Saved() == nil || s.End() != s.Saved() {
		t.Fatalf("Err %v Saved %v End %v", s.Err(), s.Saved(), s.End())
	}
}

func TestStructNameInErrors(t *testing.T) {
	s := NewScanner([]byte(`{"ID":"x"}`), (*user)(nil))
	s.Object()
	var n int
	for s.Next() {
		Int(&s, &n)
	}
	ute := s.Saved().(*stdjson.UnmarshalTypeError)
	if ute.Struct != "user" || ute.Field != "ID" || ute.Type != reflect.TypeFor[int]() {
		t.Fatalf("%#v", ute)
	}
	if want := "json: cannot unmarshal string into Go struct field user.ID of type int"; ute.Error() != want {
		t.Fatalf("Error() = %q, want %q", ute.Error(), want)
	}
}

func TestScannerValue(t *testing.T) {
	t.Run("ScalarsAndContainers", func(t *testing.T) {
		cases := []struct {
			name string
			in   string
			want string
		}{
			{"object", `{"a": 1, "b": {"c": [true, false]}}`, `{"a": 1, "b": {"c": [true, false]}}`},
			{"nested braces inside string", `{"msg": "escaped { braces } inside string", "id": 123}`, `{"msg": "escaped { braces } inside string", "id": 123}`},
			{"array", `[1, "two", {"three": 3}, [4]]`, `[1, "two", {"three": 3}, [4]]`},
			{"string with escapes", `"hello \"world\" \n \t \\ \/ \u0061"`, `"hello \"world\" \n \t \\ \/ \u0061"`},
			{"integer", `12345`, `12345`},
			{"float", `-42.5e+3`, `-42.5e+3`},
			{"bool", `true`, `true`},
			{"null", `null`, `null`},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				inBytes := []byte(tc.in)
				s := NewScanner(inBytes, (*user)(nil))
				got := s.Value()
				if s.Err() != nil {
					t.Fatalf("unexpected error: %v", s.Err())
				}
				if string(got) != tc.want {
					t.Fatalf("got %q, want %q", string(got), tc.want)
				}
				if len(got) > 0 && &got[0] != &inBytes[s.start] {
					t.Fatalf("got slice does not alias input buffer at %d", s.start)
				}
			})
		}
	})

	t.Run("EnclosingObjectIteration", func(t *testing.T) {
		data := `{"obj": {"inner": "val"}, "str": "hello {world} \"quote\"", "num": 1234, "arr": [1, "two"]}`
		s := NewScanner([]byte(data), (*user)(nil))
		if !s.Object() {
			t.Fatalf("Object failed: %v", s.Err())
		}
		wants := map[string]string{
			"obj": `{"inner": "val"}`,
			"str": `"hello {world} \"quote\""`,
			"num": `1234`,
			"arr": `[1, "two"]`,
		}
		got := make(map[string]string)
		for s.Next() {
			k := s.Key()
			v := s.Value()
			got[k] = string(v)
		}
		if err := s.End(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(got, wants) {
			t.Fatalf("got %#v, want %#v", got, wants)
		}
	})

	t.Run("TruncatedSpans", func(t *testing.T) {
		truncated := []string{
			`{"unclosed": "brace"`,
			`[1, 2,`,
			`"unclosed string`,
			`{"k":`,
			`-`,
		}
		for _, tc := range truncated {
			t.Run(tc, func(t *testing.T) {
				s := NewScanner([]byte(tc), (*user)(nil))
				got := s.Value()
				if got != nil {
					t.Fatalf("expected nil for truncated %q, got %q", tc, string(got))
				}
				if s.Err() == nil {
					t.Fatalf("expected error for truncated %q", tc)
				}
				var raw stdjson.RawMessage
				want := stdjson.Unmarshal([]byte(tc), &raw)
				if !sameError(s.Err(), want) {
					t.Fatalf("Err() = %v, want %v", s.Err(), want)
				}
			})
		}
	})
}

func refEscapeLen(b []byte) int {
	if len(b) < 2 {
		return 0
	}
	if strings.IndexByte(`"\/bfnrt`, b[1]) >= 0 {
		return 2
	}
	if b[1] != 'u' || len(b) < 6 {
		return 0
	}
	for _, h := range b[2:6] {
		if strings.IndexByte("0123456789abcdefABCDEF", h) < 0 {
			return 0
		}
	}
	return 6
}

func refScanString(data []byte, i int) (end, flags int) {
	for j := i + 1; j < len(data); {
		c := data[j]
		switch {
		case c == '"':
			return j + 1, flags
		case c < 0x20:
			return -1, 0
		case c == '\\':
			n := refEscapeLen(data[j:])
			if n == 0 {
				return -1, 0
			}
			flags |= strEscaped
			j += n
		case c >= 0x80:
			flags |= strHigh
			j++
		default:
			j++
		}
	}
	return -1, 0
}

func refEscapedRune(t testing.TB, esc []byte) rune {
	t.Helper()
	if len(esc) == 6 {
		v, err := strconv.ParseUint(string(esc[2:]), 16, 32)
		if err != nil || (v >= 0xd800 && v < 0xe000) {
			t.Fatalf("unsupported escape %q", esc)
		}
		return rune(v)
	}
	switch esc[1] {
	case 'b':
		return '\b'
	case 'f':
		return '\f'
	case 'n':
		return '\n'
	case 'r':
		return '\r'
	case 't':
		return '\t'
	}
	return rune(esc[1])
}

func refUnquote(t testing.TB, body []byte) ([]byte, bool) {
	t.Helper()
	var out []byte
	for i := 0; i < len(body); {
		c := body[i]
		switch {
		case c == '"' || c < 0x20:
			return out, false
		case c == '\\':
			n := refEscapeLen(body[i:])
			if n == 0 {
				return out, false
			}
			out = utf8.AppendRune(out, refEscapedRune(t, body[i:i+n]))
			i += n
		case c >= 0x80:
			r, size := utf8.DecodeRune(body[i:])
			out = utf8.AppendRune(out, r)
			i += size
		default:
			out = append(out, c)
			i++
		}
	}
	return out, true
}

func quotedInput(pad int, body []byte, tail string) []byte {
	data := make([]byte, pad, pad+1+len(body)+len(tail))
	data = append(data, '"')
	data = append(data, body...)
	return append(data, tail...)
}

var scanSequences = []string{
	"\x00", "\t", "\x1f", " ", "\"", "\\", "/", "<", "\x7f", "\x80", "\xbf", "\xc3", "\xff",
	"\u00e9", "\U0001f600", "\u2028", "\xe2\x80",
	`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, `\u0041`, `\u00Ff`, `\u00`, `\u00g1`, `\'`, `\a`, `\x41`,
	"\\\x00", "\\\u00e9", "\\n\u00e9", "\u00e9\\n",
}

var scanConfigs = []struct {
	pad  int
	tail string
}{
	{0, `","z"`},
	{7, ""},
	{16, `"`},
}

var unquoteSequences = []string{
	"\x00", "\x1f", "\"", "\\", "\x80", "\xbf", "\xc3", "\xff", "\u00e9", "\U0001f600", "\u2028", "\xe2\x80",
	`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, `\u0041`, `\u00e9`, `\u2028`, `\uffff`, `\u00`, `\u00g1`, `\x41`,
	"\\\x00", "\\\u00e9", "\\n\u00e9",
}

func TestScanStringRules(t *testing.T) {
	cases := []struct {
		in        string
		i         int
		wantEnd   int
		wantFlags int
	}{
		{`""`, 0, 2, 0},
		{`"abc"x`, 0, 5, 0},
		{`"a\"b"`, 0, 6, strEscaped},
		{`"a\\"`, 0, 5, strEscaped},
		{`"\/\b\f\n\r\t"`, 0, 14, strEscaped},
		{`"\u00e9\uABCD"`, 0, 14, strEscaped},
		{`"\u00e"`, 0, -1, 0},
		{`"\u00g1"`, 0, -1, 0},
		{`"\'"`, 0, -1, 0},
		{`"\x"`, 0, -1, 0},
		{`"abc`, 0, -1, 0},
		{`"abc\`, 0, -1, 0},
		{"\"a\x00\"", 0, -1, 0},
		{"\"a\x1f\"", 0, -1, 0},
		{"\"a\x7f\"", 0, 4, 0},
		{"\"\xff\"", 0, 3, strHigh},
		{"\"\xc3\xa9\\n\"", 0, 6, strHigh | strEscaped},
		{"  \"ab\"", 2, 6, 0},
		{"\"" + strings.Repeat("a", 40) + "\"", 0, 42, 0},
		{"\"" + strings.Repeat("a", 40), 0, -1, 0},
		{"\"" + strings.Repeat("a", 40) + "\\\"", 0, -1, 0},
		{"\"" + strings.Repeat("\xe2\x80\xa8", 20) + "\"", 0, 62, strHigh},
		{"\"" + strings.Repeat("a", 40) + `\n` + strings.Repeat("a", 40) + "\"", 0, 84, strEscaped},
		{"\"" + strings.Repeat("a", 40) + "\u00e9" + strings.Repeat("a", 40) + "\"", 0, 84, strHigh},
	}
	for _, c := range cases {
		data := []byte(c.in)
		if end, flags := scanString(data, c.i); end != c.wantEnd || flags != c.wantFlags {
			t.Errorf("scanString(%q, %d) = (%d, %d), want (%d, %d)", c.in, c.i, end, flags, c.wantEnd, c.wantFlags)
		}
		if end, flags := refScanString(data, c.i); end != c.wantEnd || flags != c.wantFlags {
			t.Errorf("refScanString(%q, %d) = (%d, %d), want (%d, %d)", c.in, c.i, end, flags, c.wantEnd, c.wantFlags)
		}
	}
}

func checkScanString(t *testing.T, data []byte, from int) {
	t.Helper()
	wantEnd, wantFlags := refScanString(data, from)
	gotEnd, gotFlags := scanString(data, from)
	if gotEnd != wantEnd || gotFlags != wantFlags {
		t.Fatalf("scanString(%q, %d) = (%d, %d), want (%d, %d)", data, from, gotEnd, gotFlags, wantEnd, wantFlags)
	}
}

func checkAppendUnquoted(t *testing.T, body []byte, pad int) {
	t.Helper()
	data := make([]byte, pad+len(body))
	copy(data[pad:], body)
	in := data[pad:]
	want, wantOK := refUnquote(t, in)
	got, ok := appendUnquoted(make([]byte, 0, len(in)), in)
	if ok != wantOK || !bytes.Equal(got, want) {
		t.Fatalf("appendUnquoted(%q) = (%q, %v), want (%q, %v)", in, got, ok, want, wantOK)
	}
}

var scanGapSequences = []string{`\n`, `\u00e9`, "\u00e9", `"`, "\x01", "\xff", `\x`}

var unquoteGapSequences = []string{`\n`, `\u00e9`, "\u00e9", `"`, "\x01", "\xff", `\"`}

func TestScanStringDifferential(t *testing.T) {
	forEachPlacement(scanSequences, func(body []byte) {
		for _, c := range scanConfigs {
			checkScanString(t, quotedInput(c.pad, body, c.tail), c.pad)
		}
	})
}

func TestScanStringGapDifferential(t *testing.T) {
	forEachGapInput(scanGapSequences, func(body []byte) {
		for _, c := range scanConfigs {
			checkScanString(t, quotedInput(c.pad, body, c.tail), c.pad)
		}
	})
}

func TestAppendUnquotedDifferential(t *testing.T) {
	forEachPlacement(unquoteSequences, func(body []byte) {
		for _, pad := range []int{0, 7} {
			checkAppendUnquoted(t, body, pad)
		}
	})
}

func TestAppendUnquotedGapDifferential(t *testing.T) {
	forEachGapInput(unquoteGapSequences, func(body []byte) {
		checkAppendUnquoted(t, body, 0)
	})
}
