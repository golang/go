// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	stdjson "encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

type walkGenT struct {
	Name   string    `json:"name"`
	Next   *walkGenT `json:"next,omitempty"`
	Kids   []any     `json:"kids,omitempty"`
	Skip   string    `json:"-"`
	hidden string
}

func (walkGenT) MarshalJSON() ([]byte, error) {
	return []byte(`"honoured"`), nil
}

type walkUserT struct {
	V int `json:"v"`
}

func (walkUserT) MarshalJSON() ([]byte, error) {
	return []byte(`"user"`), nil
}

type walkTextT struct {
	V int `json:"v"`
}

func (walkTextT) MarshalText() ([]byte, error) {
	return []byte("text"), nil
}

type walkEmbed struct {
	E string `json:"e"`
}

type walkOuter struct {
	walkEmbed
	Name string `json:"name"`
	Skip int    `json:"-"`
	Omit []int  `json:"omit,omitempty"`
	Zero string `json:"zero,omitzero"`
	QS   int    `json:"qs,string"`
	HTML string `json:"html"`
}

func walkGenWalk() Walk {
	var s Walk
	s.AddType(reflect.TypeOf(walkGenT{}))
	return s
}

func TestWalkCycleTerminates(t *testing.T) {
	n := &walkGenT{Name: "a"}
	n.Next = n
	s := walkGenWalk()
	_, err := s.Marshal(n)
	if err == nil {
		t.Fatal("self cycle encoded without an error")
	}
	uerr, ok := err.(*stdjson.UnsupportedValueError)
	if !ok || !strings.HasPrefix(uerr.Str, "encountered a cycle via ") {
		t.Fatalf("cycle error = %#v, want UnsupportedValueError with cycle message", err)
	}
	m := map[string]any{"n": map[string]any{"self": nil}}
	m["n"].(map[string]any)["self"] = m
	var s2 Walk
	if _, err := s2.Marshal(m); err == nil {
		t.Fatal("map cycle encoded without an error")
	} else if !strings.HasPrefix(err.Error(), "json: unsupported value: encountered a cycle via ") {
		t.Fatalf("map cycle error = %q", err.Error())
	}
}

func TestWalkGenDispatchWalks(t *testing.T) {
	v := walkGenT{Name: "a", Next: &walkGenT{Name: "b"}}
	s := walkGenWalk()
	got, err := s.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(got), "honoured") {
		t.Fatalf("generated-side type was honoured instead of walked: %s", got)
	}
	var s2 Walk
	got2, err := s2.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if string(got2) != `"honoured"` {
		t.Fatalf("user-method type outside Gen was not honoured: %s", got2)
	}
}

type walkBothT struct {
	V int `json:"v"`
}

func (walkBothT) MarshalText() ([]byte, error) {
	return []byte("text"), nil
}

func (*walkBothT) MarshalJSON() ([]byte, error) {
	return []byte(`"ptr"`), nil
}

func TestWalkAddressabilityPrecedence(t *testing.T) {
	vals := []any{
		walkBothT{V: 1},
		[]walkBothT{{V: 2}},
		struct {
			B walkBothT `json:"b"`
		}{B: walkBothT{V: 3}},
	}
	wantBody := []string{`"text"`, `["ptr"]`, `{"b":"text"}`}
	for i, v := range vals {
		var s Walk
		got, gerr := s.Marshal(v)
		want, werr := stdjson.Marshal(v)
		if (gerr == nil) != (werr == nil) {
			t.Fatalf("case %d: walk error %v, reflective error %v", i, gerr, werr)
		}
		if gerr != nil {
			if gerr.Error() != werr.Error() {
				t.Fatalf("case %d: walk error %q, reflective error %q", i, gerr.Error(), werr.Error())
			}
			continue
		}
		if string(got) != string(want) {
			t.Fatalf("case %d: walk %s, reflective %s", i, got, want)
		}
		if string(got) != wantBody[i] {
			t.Fatalf("case %d: got %s, want %s", i, got, wantBody[i])
		}
	}
}

func TestWalkHonoursUserMethods(t *testing.T) {
	var s Walk
	got, err := s.Marshal(walkUserT{V: 1})
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != `"user"` {
		t.Fatalf("user MarshalJSON lost: %s", got)
	}
	got, err = s.Marshal(walkTextT{V: 1})
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != `"text"` {
		t.Fatalf("user MarshalText lost: %s", got)
	}
	got, err = s.Marshal(struct {
		U walkUserT `json:"u"`
	}{U: walkUserT{V: 2}})
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != `{"u":"user"}` {
		t.Fatalf("nested user method lost: %s", got)
	}
}

func TestWalkParitySweep(t *testing.T) {
	ts := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	vals := []any{
		nil,
		true,
		-42,
		uint64(1 << 63),
		2.5,
		"a<b>&\u2028\"q\"\\",
		[]byte("bytes"),
		[]int{3, 1, 2},
		[]any{"s", 1.5, true, nil},
		[]string{},
		[]int(nil),
		map[string]int{"b": 2, "a": 1},
		map[int]string{10: "j", 2: "b"},
		map[string]any(nil),
		map[string]any{},
		walkOuter{walkEmbed: walkEmbed{E: "e"}, Name: "n", Omit: nil, Zero: "", QS: 7, HTML: "<>&"},
		walkOuter{walkEmbed: walkEmbed{E: "e"}, Name: "n", Omit: []int{1}, Zero: "z", QS: 0, HTML: "x"},
		struct {
			A   int       `json:"a,string"`
			B   bool      `json:"b,string"`
			S   string    `json:"s,string"`
			F   float64   `json:"f,string"`
			M   walkTextT `json:"m"`
			T   time.Time `json:"t"`
			P   *int      `json:"p"`
			Any any       `json:"any"`
			Arr [2]int    `json:"arr"`
		}{A: 1, B: true, S: "s", F: 0.5, M: walkTextT{V: 1}, T: ts, Any: []int{9}},
		[2]map[string]int{{"a": 1}, nil},
	}
	for i, v := range vals {
		var s Walk
		got, gerr := s.Marshal(v)
		want, werr := stdjson.Marshal(v)
		if (gerr == nil) != (werr == nil) {
			t.Fatalf("case %d (%T): walk error %v, reflective error %v", i, v, gerr, werr)
		}
		if gerr != nil {
			if gerr.Error() != werr.Error() {
				t.Fatalf("case %d (%T): walk error %q, reflective error %q", i, v, gerr.Error(), werr.Error())
			}
			continue
		}
		if string(got) != string(want) {
			t.Fatalf("case %d (%T): walk %s, reflective %s", i, v, got, want)
		}
	}
}

func checkWalkParity(t *testing.T, label string, v any) {
	t.Helper()
	var s Walk
	got, gerr := s.Marshal(v)
	want, werr := stdjson.Marshal(v)
	if (gerr == nil) != (werr == nil) {
		t.Fatalf("%s: walk error %v, oracle error %v", label, gerr, werr)
	}
	if gerr != nil {
		if gerr.Error() != werr.Error() {
			t.Fatalf("%s: walk error %q, oracle error %q", label, gerr.Error(), werr.Error())
		}
		if reflect.TypeOf(gerr) != reflect.TypeOf(werr) {
			t.Fatalf("%s: walk error type %T, oracle error type %T", label, gerr, werr)
		}
		return
	}
	if string(got) != string(want) {
		t.Fatalf("%s: walk %s, oracle %s", label, got, want)
	}
}

func TestWalkStringOptionQuoted(t *testing.T) {
	i := 42
	str := "a<b>&"
	checkWalkParity(t, "int,string", struct {
		I int `json:"i,string"`
	}{I: 42})
	checkWalkParity(t, "ptr int,string", struct {
		I *int `json:"i,string"`
	}{I: &i})
	checkWalkParity(t, "nil ptr int,string", struct {
		I *int `json:"i,string"`
	}{})
	checkWalkParity(t, "uint,string", struct {
		U uint64 `json:"u,string"`
	}{U: 1 << 63})
	checkWalkParity(t, "string,string", struct {
		S string `json:"s,string"`
	}{S: str})
	checkWalkParity(t, "ptr string,string", struct {
		S *string `json:"s,string"`
	}{S: &str})
	checkWalkParity(t, "bool,string", struct {
		B bool `json:"b,string"`
	}{B: true})
	checkWalkParity(t, "float,string", struct {
		F float64 `json:"f,string"`
	}{F: 0.5})
}

func TestWalkStringQuoting(t *testing.T) {
	checkWalkParity(t, "plain string", struct {
		S string `json:"s"`
	}{S: "a<b>&\"\u2028\\"})
	checkWalkParity(t, "nested strings", map[string]any{
		"k": []any{"x&y", ""},
	})
}

func TestWalkDiamondDuplicate(t *testing.T) {
	type A struct {
		X int `json:"x"`
	}
	type B struct {
		A
	}
	type C struct {
		A
	}
	dTyp := reflect.StructOf([]reflect.StructField{
		{Name: "B", Type: reflect.TypeOf(B{}), Anonymous: true},
		{Name: "C", Type: reflect.TypeOf(C{}), Anonymous: true},
	})
	d := reflect.New(dTyp).Elem()
	d.Field(0).Set(reflect.ValueOf(B{A: A{X: 1}}))
	d.Field(1).Set(reflect.ValueOf(C{A: A{X: 2}}))
	checkWalkParity(t, "diamond", d.Interface())
}

func TestWalkNilMapBadKey(t *testing.T) {
	var m map[struct{}]int
	var s Walk
	got, gerr := s.Marshal(m)
	want, werr := stdjson.Marshal(m)
	if (gerr == nil) != (werr == nil) {
		t.Fatalf("nil bad-key map: walk error %v, oracle error %v", gerr, werr)
	}
	if gerr != nil {
		if _, ok := gerr.(*stdjson.UnsupportedTypeError); stdIsV2 == false && !ok {
			t.Fatalf("v1 nil bad-key map: walk error type %T, want UnsupportedTypeError", gerr)
		}
		if gerr.Error() != werr.Error() {
			t.Fatalf("nil bad-key map: walk error %q, oracle error %q", gerr.Error(), werr.Error())
		}
		return
	}
	if string(got) != string(want) {
		t.Fatalf("nil bad-key map: walk %s, oracle %s", got, want)
	}
}

type walkZeroT struct{}

func (walkZeroT) IsZero() bool { return true }

func TestWalkOmitZeroNilPointer(t *testing.T) {
	checkWalkParity(t, "nil ptr omitzero", struct {
		P *walkZeroT `json:"p,omitzero"`
	}{})
	checkWalkParity(t, "zero ptr omitzero", struct {
		P *walkZeroT `json:"p,omitzero"`
	}{P: &walkZeroT{}})
	type I int
	checkWalkParity(t, "nil embedded-equivalent omitzero", struct {
		P *I `json:"p,omitzero"`
	}{})
}

func TestRebaseLeavesUserSyntaxErrorUntouched(t *testing.T) {
	var v any
	err := stdjson.Unmarshal([]byte("{"), &v)
	user := err.(*stdjson.SyntaxError)
	before := *user
	var s Walk
	if got := s.RebaseDepthError(user, 71); got != user {
		t.Fatalf("user error identity changed: %p vs %p", got, user)
	}
	if user.Offset != before.Offset || user.Error() != before.Error() {
		t.Fatalf("user error was rebased: %#v vs %#v", user, &before)
	}
}

func TestWalkUnsupportedKinds(t *testing.T) {
	var s Walk
	if _, err := s.Marshal(func() {}); err == nil {
		t.Fatal("func value encoded without an error")
	} else if _, ok := err.(*stdjson.UnsupportedTypeError); !ok {
		t.Fatalf("func value error = %#v, want UnsupportedTypeError", err)
	}
	if _, err := s.Marshal(map[bool]int{true: 1}); err == nil {
		t.Fatal("bool-keyed map encoded without an error")
	} else if _, ok := err.(*stdjson.UnsupportedTypeError); !ok {
		t.Fatalf("bool-keyed map error = %#v, want UnsupportedTypeError", err)
	}
	if _, err := s.Marshal(complex(1, 2)); err == nil {
		t.Fatal("complex value encoded without an error")
	} else if _, ok := err.(*stdjson.UnsupportedTypeError); !ok {
		t.Fatalf("complex value error = %#v, want UnsupportedTypeError", err)
	}
}

func TestWalkDeepNestingMatchesCap(t *testing.T) {
	type link struct {
		Next *link `json:"next,omitempty"`
	}
	for _, n := range []int{3000, 9000, 20000} {
		head := &link{}
		cur := head
		for range n {
			cur.Next = &link{}
			cur = cur.Next
		}
		var s Walk
		got, gerr := s.Marshal(head)
		want, werr := stdjson.Marshal(head)
		if (gerr == nil) != (werr == nil) {
			t.Fatalf("depth %d: walk error %v, oracle error %v", n, gerr, werr)
		}
		if gerr != nil {
			if gerr.Error() != werr.Error() {
				t.Fatalf("depth %d: walk error %q, oracle error %q", n, gerr.Error(), werr.Error())
			}
			if stdIsV2 {
				if _, ok := gerr.(*stdjson.SyntaxError); !ok {
					t.Fatalf("depth %d: walk error type %T, want *json.SyntaxError", n, gerr)
				}
			}
			continue
		}
		if string(got) != string(want) {
			t.Fatalf("depth %d mismatches oracle", n)
		}
	}
}

func TestWalkDAGSharesState(t *testing.T) {
	shared := &walkGenT{Name: "s"}
	v := struct {
		L *walkGenT `json:"l"`
		R *walkGenT `json:"r"`
	}{L: shared, R: shared}
	var s Walk
	s.AddType(reflect.TypeOf(walkGenT{}))
	got, err := s.Marshal(v)
	if err != nil {
		t.Fatalf("shared tail reported a cycle: %v", err)
	}
	if strings.Count(string(got), `"s"`) != 2 {
		t.Fatalf("shared tail encoded once: %s", got)
	}
}
