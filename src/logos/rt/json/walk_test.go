// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	stdjson "encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"
)

func checkAnyParity(t *testing.T, label string, v any) {
	t.Helper()
	var s Walk
	got, gerr := s.AppendAny(nil, v)
	want, werr := stdjson.Marshal(v)
	if (gerr == nil) != (werr == nil) {
		t.Fatalf("%s: walk error %v, encoding/json error %v", label, gerr, werr)
	}
	if gerr != nil {
		if gerr.Error() != werr.Error() || reflect.TypeOf(gerr) != reflect.TypeOf(werr) {
			t.Fatalf("%s: walk error %T %q, encoding/json error %T %q", label, gerr, gerr.Error(), werr, werr.Error())
		}
		return
	}
	if string(got) != string(want) {
		t.Fatalf("%s: walk %s, encoding/json %s", label, got, want)
	}
}

type walkNode struct {
	Name string    `json:"name"`
	Next *walkNode `json:"next,omitempty"`
	Kids []any     `json:"kids,omitempty"`
}

func cyclePrefixed(err error) bool {
	_, ok := err.(*stdjson.UnsupportedValueError)
	return ok && strings.HasPrefix(err.Error(), "json: unsupported value: encountered a cycle via ")
}

func TestWalkCycleTerminates(t *testing.T) {
	n := &walkNode{Name: "a"}
	n.Next = n
	m := map[string]any{"n": map[string]any{"self": nil}}
	m["n"].(map[string]any)["self"] = m
	k := &walkNode{Name: "k"}
	k.Kids = []any{[]any{k}}
	for _, v := range []any{n, m, k} {
		var s Walk
		_, err := s.AppendAny(nil, v)
		_, werr := stdjson.Marshal(v)
		if !cyclePrefixed(err) || !cyclePrefixed(werr) {
			t.Fatalf("cycle %T: walk error %v, encoding/json error %v", v, err, werr)
		}
	}
}

func TestWalkGenDispatchSharesState(t *testing.T) {
	n := &walkNode{Name: "g"}
	n.Kids = []any{map[string]any{"back": n}}
	var s Walk
	calls := 0
	s.Gen = func(s *Walk, buf []byte, v any) ([]byte, bool, error) {
		p, ok := v.(*walkNode)
		if !ok {
			return buf, false, nil
		}
		calls++
		if err := EnterPointer(s, p); err != nil {
			return buf, true, err
		}
		defer LeavePointer(s, p)
		out, err := s.AppendAny(buf, p.Kids)
		return out, true, err
	}
	_, err := s.AppendAny(nil, n)
	if !cyclePrefixed(err) {
		t.Fatalf("cycle through Gen and a dynamic map = %v, want the encoding/json cycle error", err)
	}
	if calls < 2 {
		t.Fatalf("Gen ran %d times, want the cycle found through it", calls)
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
		&struct {
			B walkBothT `json:"b"`
		}{B: walkBothT{V: 4}},
	}
	bodies := []string{`"text"`, `["ptr"]`, `{"b":"text"}`, `{"b":"ptr"}`}
	for i, v := range vals {
		checkAnyParity(t, bodies[i], v)
		var s Walk
		if got, _ := s.AppendAny(nil, v); string(got) != bodies[i] {
			t.Fatalf("case %d: got %s, want %s", i, got, bodies[i])
		}
	}
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

func TestWalkParitySweep(t *testing.T) {
	ts := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	i, str := 42, "a<b>&"
	vals := []any{
		nil, true, -42, uint64(1 << 63), 2.5, "a<b>&\u2028\"q\"\\",
		[]byte("bytes"), []int{3, 1, 2}, []any{"s", 1.5, true, nil}, []string{}, []int(nil),
		map[string]int{"b": 2, "a": 1}, map[int]string{10: "j", 2: "b"}, map[string]any(nil), map[string]any{},
		walkUserT{V: 1}, walkTextT{V: 1}, struct {
			U walkUserT `json:"u"`
		}{U: walkUserT{V: 2}},
		walkOuter{E: "e", Name: "n", QS: 7, HTML: "<>&"},
		walkOuter{E: "e", Name: "n", Omit: []int{1}, Zero: "z", HTML: "x"},
		struct {
			A   int       `json:"a,string"`
			B   bool      `json:"b,string"`
			S   string    `json:"s,string"`
			F   float64   `json:"f,string"`
			PI  *int      `json:"pi,string"`
			PS  *string   `json:"ps,string"`
			NI  *int      `json:"ni,string"`
			M   walkTextT `json:"m"`
			T   time.Time `json:"t"`
			Any any       `json:"any"`
			Arr [2]int    `json:"arr"`
		}{A: 1, B: true, S: str, F: 0.5, PI: &i, PS: &str, M: walkTextT{V: 1}, T: ts, Any: []int{9}},
		[2]map[string]int{{"a": 1}, nil},
		map[string]any{"k": []any{"x&y", ""}},
		stdjson.Number("12"), []stdjson.Number{"1", ""}, map[string]stdjson.Number{"a": "7"},
		struct {
			N  stdjson.Number  `json:"n,string"`
			PN *stdjson.Number `json:"pn,string"`
		}{N: "5"},
	}
	for i, v := range vals {
		checkAnyParity(t, fmt.Sprintf("sweep %d (%T)", i, v), v)
	}
}

type walkZeroT struct{}

func (walkZeroT) IsZero() bool { return true }

func TestWalkOmitZero(t *testing.T) {
	type I int
	checkAnyParity(t, "nil ptr omitzero", struct {
		P *walkZeroT `json:"p,omitzero"`
	}{})
	checkAnyParity(t, "zero ptr omitzero", struct {
		P *walkZeroT `json:"p,omitzero"`
	}{P: &walkZeroT{}})
	checkAnyParity(t, "nil named ptr omitzero", struct {
		P *I `json:"p,omitzero"`
	}{})
	checkAnyParity(t, "interface omitzero", struct {
		Z interface{ IsZero() bool } `json:"z,omitzero"`
	}{Z: (*walkZeroT)(nil)})
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
	checkAnyParity(t, "diamond", d.Interface())
}

func TestWalkUnsupportedKinds(t *testing.T) {
	var badKey map[struct{}]int
	for _, v := range []any{func() {}, map[bool]int{true: 1}, complex(1, 2), badKey, []chan int{make(chan int)}} {
		checkAnyParity(t, reflect.TypeOf(v).String(), v)
	}
}

func checkDepthParity(t *testing.T, label string, v any) {
	t.Helper()
	var s Walk
	_, gerr := s.AppendAny(nil, v)
	_, werr := stdjson.Marshal(v)
	ge, gok := gerr.(*stdjson.SyntaxError)
	we, wok := werr.(*stdjson.SyntaxError)
	if (gerr == nil) != (werr == nil) || gok != wok || (gok && (ge.Error() != we.Error() || ge.Offset != we.Offset)) {
		t.Fatalf("%s: walk error %#v, encoding/json error %#v", label, gerr, werr)
	}
}

func TestWalkDeepNestingMatchesCap(t *testing.T) {
	type link struct {
		Next *link `json:"next,omitempty"`
	}
	for _, n := range []int{3000, 9998, 9999, 10000, 20000} {
		head := &link{}
		cur := head
		for range n {
			cur.Next = &link{}
			cur = cur.Next
		}
		checkDepthParity(t, fmt.Sprintf("depth %d", n), head)
	}
}

func TestWalkEmptyContainerOpensNoFrame(t *testing.T) {
	tails := []any{[]any{}, map[string]any{}, []int{}, map[string]int{}, []any{1}, map[string]any{"k": 1}}
	for _, tail := range tails {
		for _, n := range []int{9998, 9999, 10000} {
			v := tail
			for range n {
				v = []any{v}
			}
			checkDepthParity(t, fmt.Sprintf("%#v under %d arrays", tail, n), v)
		}
	}
}

func TestWalkDAGSharesState(t *testing.T) {
	shared := &walkNode{Name: "s"}
	checkAnyParity(t, "dag", struct {
		L *walkNode `json:"l"`
		R *walkNode `json:"r"`
	}{L: shared, R: shared})
}
