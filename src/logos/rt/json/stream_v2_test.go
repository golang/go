// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	stdjson "encoding/json"
	"encoding/json/jsontext"
	"errors"
	"reflect"
	"strings"
	"testing"
)

type streamProbe string

func (p *streamProbe) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	tok, err := dec.ReadValue()
	if err != nil {
		return err
	}
	*p += streamProbe("|from:" + string(tok))
	return nil
}

func (p *streamProbe) UnmarshalJSON(b []byte) error {
	*p += streamProbe("|json:" + string(b))
	return nil
}

func (p *streamProbe) UnmarshalText(b []byte) error {
	*p += streamProbe("|text:" + string(b))
	return nil
}

type streamFailProbe string

func (p *streamFailProbe) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	tok, err := dec.ReadValue()
	if err != nil {
		return err
	}
	*p += streamFailProbe("|from:" + string(tok))
	return &stdjson.UnmarshalTypeError{Value: string(tok), Type: reflect.TypeFor[streamFailProbe]()}
}

func (p *streamFailProbe) UnmarshalJSON(b []byte) error {
	*p += streamFailProbe("|json:" + string(b))
	return nil
}

func (p *streamFailProbe) UnmarshalText(b []byte) error {
	*p += streamFailProbe("|text:" + string(b))
	return nil
}

type streamTarget[T any] struct {
	V     T   `json:"v,string"`
	After int `json:"after"`
}

func checkStream[T any](t *testing.T, in string, newDst func(*streamTarget[T]) any) {
	t.Helper()
	data := `{"v":` + in + `,"after":17}`
	var got, want streamTarget[T]
	s := NewScanner([]byte(data), (*streamTarget[T])(nil))
	if !s.Object() {
		t.Fatalf("%s: Object failed: %v", data, s.Err())
	}
	for s.Next() {
		switch s.Key() {
		case "v":
			switch dst := newDst(&got).(type) {
			case *streamProbe:
				QuotedFrom[streamProbe](&s, dst)
			case **streamProbe:
				QuotedFromPtr[streamProbe](&s, dst)
			case *streamFailProbe:
				QuotedFrom[streamFailProbe](&s, dst)
			case **streamFailProbe:
				QuotedFromPtr[streamFailProbe](&s, dst)
			default:
				t.Fatalf("%s: unsupported stream target %T", data, dst)
			}
		case "after":
			Int(&s, &got.After)
		}
	}
	wantErr := stdjson.Unmarshal([]byte(data), &want)
	if !sameError(s.End(), wantErr) {
		t.Errorf("%s: End() = %v, encoding/json = %v", in, s.End(), wantErr)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("%s: value = %#v, encoding/json = %#v", in, got, want)
	}
}

func TestStreamPrecedenceMatchesEncodingJSON(t *testing.T) {
	inputs := []string{`"\"hi\""`, `"3"`, `null`, `"null"`, `7`, `true`, `{}`, `""`}
	for _, in := range inputs {
		checkStream[streamProbe](t, in, func(g *streamTarget[streamProbe]) any { return &g.V })
		checkStream[*streamProbe](t, in, func(g *streamTarget[*streamProbe]) any { return &g.V })
	}
}

func TestStreamErrorContinuationMatchesEncodingJSON(t *testing.T) {
	inputs := []string{`"\"hi\""`, `"3"`, `null`, `"null"`, `7`, `""`}
	for _, in := range inputs {
		checkStream[streamFailProbe](t, in, func(g *streamTarget[streamFailProbe]) any { return &g.V })
		checkStream[*streamFailProbe](t, in, func(g *streamTarget[*streamFailProbe]) any { return &g.V })
	}
}

type streamUnsupportedJSONProbe string

func (p *streamUnsupportedJSONProbe) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	return errors.ErrUnsupported
}

func (p *streamUnsupportedJSONProbe) UnmarshalJSON(b []byte) error {
	*p += streamUnsupportedJSONProbe("|json:" + string(b))
	return nil
}

type streamUnsupportedTextProbe string

func (p *streamUnsupportedTextProbe) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	return errors.ErrUnsupported
}

func (p *streamUnsupportedTextProbe) UnmarshalText(b []byte) error {
	*p += streamUnsupportedTextProbe("|text:" + string(b))
	return nil
}

type streamUnsupportedMutatedProbe string

func (p *streamUnsupportedMutatedProbe) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	if _, err := dec.ReadValue(); err != nil {
		return err
	}
	return errors.ErrUnsupported
}

func (p *streamUnsupportedMutatedProbe) UnmarshalJSON(b []byte) error {
	*p += streamUnsupportedMutatedProbe("|json:" + string(b))
	return nil
}

type streamNoConsumeProbe string

func (p *streamNoConsumeProbe) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	return nil
}

// streamDeclineTextProbe declines its first streaming call; a fallback
// decode that dispatches the method a second time changes the recorded
// calls and the decoded value, so encoding/json divergence is visible.
type streamDeclineTextProbe struct {
	Calls int
	Text  string
}

func (p *streamDeclineTextProbe) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	p.Calls++
	if p.Calls == 1 {
		return errors.ErrUnsupported
	}
	tok, err := dec.ReadValue()
	if err != nil {
		return err
	}
	p.Text += "|from:" + string(tok)
	return nil
}

func (p *streamDeclineTextProbe) UnmarshalText(b []byte) error {
	p.Text += "|text:" + string(b)
	return nil
}

// streamDeclineOnlyProbe has no other unmarshal method: its fallback is
// the plain decode, which must not re-dispatch the declined method.
type streamDeclineOnlyProbe struct {
	Calls int
	Text  string
}

func (p *streamDeclineOnlyProbe) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	p.Calls++
	if p.Calls == 1 {
		return errors.ErrUnsupported
	}
	tok, err := dec.ReadValue()
	if err != nil {
		return err
	}
	p.Text += "|from:" + string(tok)
	return nil
}

func checkStreamFallback[T any](t *testing.T, in string, newDst func(*streamTarget[T]) any) {
	t.Helper()
	data := `{"v":` + in + `,"after":17}`
	var got, want streamTarget[T]
	s := NewScanner([]byte(data), (*streamTarget[T])(nil))
	if !s.Object() {
		t.Fatalf("%s: Object failed: %v", data, s.Err())
	}
	for s.Next() {
		switch s.Key() {
		case "v":
			switch dst := newDst(&got).(type) {
			case *streamUnsupportedJSONProbe:
				if !QuotedFrom[streamUnsupportedJSONProbe](&s, dst) {
					QuotedJSON(&s, dst)
				}
			case **streamUnsupportedJSONProbe:
				if !QuotedFromPtr[streamUnsupportedJSONProbe](&s, dst) {
					QuotedJSONPtr[streamUnsupportedJSONProbe](&s, dst)
				}
			case *streamUnsupportedTextProbe:
				if !QuotedFrom[streamUnsupportedTextProbe](&s, dst) {
					QuotedText(&s, dst)
				}
			case **streamUnsupportedTextProbe:
				if !QuotedFromPtr[streamUnsupportedTextProbe](&s, dst) {
					QuotedTextPtr[streamUnsupportedTextProbe](&s, dst)
				}
			case *streamUnsupportedMutatedProbe:
				QuotedFrom[streamUnsupportedMutatedProbe](&s, dst)
			case **streamUnsupportedMutatedProbe:
				QuotedFromPtr[streamUnsupportedMutatedProbe](&s, dst)
			case *streamNoConsumeProbe:
				QuotedFrom[streamNoConsumeProbe](&s, dst)
			case **streamNoConsumeProbe:
				QuotedFromPtr[streamNoConsumeProbe](&s, dst)
			case *streamDeclineTextProbe:
				if !QuotedFrom[streamDeclineTextProbe](&s, dst) {
					QuotedText(&s, dst)
				}
			case **streamDeclineTextProbe:
				if !QuotedFromPtr[streamDeclineTextProbe](&s, dst) {
					QuotedTextPtr[streamDeclineTextProbe](&s, dst)
				}
			case *streamDeclineOnlyProbe:
				if !QuotedFrom[streamDeclineOnlyProbe](&s, dst) {
					FallbackFrom(&s, dst)
				}
			case **streamDeclineOnlyProbe:
				if !QuotedFromPtr[streamDeclineOnlyProbe](&s, dst) {
					FallbackFromPtr(&s, dst)
				}
			default:
				t.Fatalf("%s: unsupported fallback target %T", data, dst)
			}
		case "after":
			Int(&s, &got.After)
		}
	}
	wantErr := stdjson.Unmarshal([]byte(data), &want)
	if !sameError(s.End(), wantErr) {
		t.Errorf("%s: End() = %v, encoding/json = %v", in, s.End(), wantErr)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("%s: value = %#v, encoding/json = %#v", in, got, want)
	}
}

func TestStreamUnsupportedFallbackMatchesEncodingJSON(t *testing.T) {
	inputs := []string{`"\"hi\""`, `"3"`, `7`, `""`}
	for _, in := range inputs {
		checkStreamFallback[streamUnsupportedJSONProbe](t, in, func(g *streamTarget[streamUnsupportedJSONProbe]) any { return &g.V })
		checkStreamFallback[*streamUnsupportedJSONProbe](t, in, func(g *streamTarget[*streamUnsupportedJSONProbe]) any { return &g.V })
		checkStreamFallback[streamUnsupportedTextProbe](t, in, func(g *streamTarget[streamUnsupportedTextProbe]) any { return &g.V })
		checkStreamFallback[*streamUnsupportedTextProbe](t, in, func(g *streamTarget[*streamUnsupportedTextProbe]) any { return &g.V })
	}
}

func TestStreamDeclineFallbackMatchesEncodingJSON(t *testing.T) {
	for _, in := range []string{`"x"`, `7`, `{}`, `[1]`, `true`, `null`, `""`} {
		checkStreamFallback[streamDeclineTextProbe](t, in, func(g *streamTarget[streamDeclineTextProbe]) any { return &g.V })
		checkStreamFallback[*streamDeclineTextProbe](t, in, func(g *streamTarget[*streamDeclineTextProbe]) any { return &g.V })
	}
	for _, in := range []string{`7`, `{}`, `[1]`, `true`, `null`} {
		checkStreamFallback[streamDeclineOnlyProbe](t, in, func(g *streamTarget[streamDeclineOnlyProbe]) any { return &g.V })
		checkStreamFallback[*streamDeclineOnlyProbe](t, in, func(g *streamTarget[*streamDeclineOnlyProbe]) any { return &g.V })
	}
}

// encoding/json unquotes `,string` method input permissively: unpaired
// surrogates and malformed UTF-8 decode as U+FFFD rather than rejecting the
// string token.
func TestQuotedTextPermissiveUnquoteMatchesEncodingJSON(t *testing.T) {
	for _, in := range []string{`"\ud800"`, `"\ud83d\ude00"`, "\"\xff\"", `"x\ud800y"`} {
		checkStreamFallback[streamUnsupportedTextProbe](t, in, func(g *streamTarget[streamUnsupportedTextProbe]) any { return &g.V })
		checkStreamFallback[*streamUnsupportedTextProbe](t, in, func(g *streamTarget[*streamUnsupportedTextProbe]) any { return &g.V })
		checkStreamFallback[streamDeclineTextProbe](t, in, func(g *streamTarget[streamDeclineTextProbe]) any { return &g.V })
	}
}

func TestStreamProtocolErrorMatchesEncodingJSON(t *testing.T) {
	inputs := []string{`"\"hi\""`, `"3"`, `7`, `{}`, `[]`, `""`}
	for _, in := range inputs {
		checkStreamFallback[streamUnsupportedMutatedProbe](t, in, func(g *streamTarget[streamUnsupportedMutatedProbe]) any { return &g.V })
		checkStreamFallback[*streamUnsupportedMutatedProbe](t, in, func(g *streamTarget[*streamUnsupportedMutatedProbe]) any { return &g.V })
		checkStreamFallback[streamNoConsumeProbe](t, in, func(g *streamTarget[streamNoConsumeProbe]) any { return &g.V })
		checkStreamFallback[*streamNoConsumeProbe](t, in, func(g *streamTarget[*streamNoConsumeProbe]) any { return &g.V })
	}
}

// streamDeclinePartialProbe decodes fields of a struct through the
// methodless fallback: valid writes must survive a later field error,
// matching encoding/json's non-transactional decode.
type streamDeclinePartialProbe struct {
	Calls int `json:"-"`
	A     int
	B     int
}

func (p *streamDeclinePartialProbe) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	p.Calls++
	return errors.ErrUnsupported
}

func TestFallbackFromKeepsPartialUpdates(t *testing.T) {
	data := `{"v":{"A":1,"B":"bad"},"after":17}`
	var got, want streamTarget[streamDeclinePartialProbe]
	s := NewScanner([]byte(data), (*streamTarget[streamDeclinePartialProbe])(nil))
	s.Object()
	for s.Next() {
		switch s.Key() {
		case "v":
			if !QuotedFrom[streamDeclinePartialProbe](&s, &got.V) {
				FallbackFrom(&s, &got.V)
			}
		case "after":
			Int(&s, &got.After)
		}
	}
	wantErr := stdjson.Unmarshal([]byte(data), &want)
	if !sameError(s.End(), wantErr) {
		t.Fatalf("End() = %v, encoding/json = %v", s.End(), wantErr)
	}
	if got.V.A != 1 || got.V.B != 0 {
		t.Fatalf("partial decode lost: got %#v", got.V)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("value = %#v, encoding/json = %#v", got, want)
	}
}

func TestFallbackFromPtrPreservesExistingAlias(t *testing.T) {
	data := `{"v":{"Text":"x"},"after":17}`
	shared := &streamDeclineOnlyProbe{Text: "seed"}
	alias := shared
	var got struct {
		V     *streamDeclineOnlyProbe `json:"v"`
		After int                     `json:"after"`
	}
	got.V = shared
	s := NewScanner([]byte(data), (*streamDeclineOnlyProbe)(nil))
	s.Object()
	for s.Next() {
		switch s.Key() {
		case "v":
			if !QuotedFromPtr[streamDeclineOnlyProbe](&s, &got.V) {
				FallbackFromPtr[streamDeclineOnlyProbe](&s, &got.V)
			}
		case "after":
			Int(&s, &got.After)
		}
	}
	if err := s.End(); err != nil {
		t.Fatal(err)
	}
	if got.V != alias {
		t.Fatalf("pointer replaced: got %p, alias %p", got.V, alias)
	}
	if alias.Text != "x" {
		t.Fatalf("alias did not observe update: Text = %q", alias.Text)
	}
}

type quotedUnsupportedJSONProbe struct{}

func (p *quotedUnsupportedJSONProbe) UnmarshalJSON(b []byte) error {
	return errors.ErrUnsupported
}

func TestQuotedJSONUnsupportedMethodIsMisuse(t *testing.T) {
	data := `{"v":"x","after":17}`
	var got struct {
		V     quotedUnsupportedJSONProbe `json:"v"`
		After int                        `json:"after"`
	}
	s := NewScanner([]byte(data), nil)
	s.Object()
	for s.Next() {
		switch s.Key() {
		case "v":
			QuotedJSON(&s, &got.V)
		case "after":
			Int(&s, &got.After)
		}
	}
	err := s.End()
	if err == nil || !strings.Contains(err.Error(), "UnmarshalJSON method may not return errors.ErrUnsupported") {
		t.Fatalf("End() = %v", err)
	}
}

type quotedAppendProbe struct {
	Token []byte
}

func (p *quotedAppendProbe) UnmarshalJSON(b []byte) error {
	p.Token = append(b, 'x')
	return nil
}

func TestQuotedJSONMethodInputIsCapped(t *testing.T) {
	data := `{"v":"7","after":17}`
	var got struct {
		V     quotedAppendProbe `json:"v"`
		After int               `json:"after"`
	}
	s := NewScanner([]byte(data), nil)
	s.Object()
	for s.Next() {
		switch s.Key() {
		case "v":
			QuotedJSON(&s, &got.V)
		case "after":
			Int(&s, &got.After)
		}
	}
	if err := s.End(); err != nil {
		t.Fatal(err)
	}
	if string(got.V.Token) != `"7"x` || got.After != 17 {
		t.Fatalf("got %#v after %d", got.V, got.After)
	}
}

// streamDeclineEmbedProbe has an unexported embedded field, a shape
// reflect.StructOf cannot rebuild without its methods. The fallback
// decodes through the type itself instead: the streaming method sees
// one more decline, and the promoted fields still decode.
type streamDeclineEmbedInner struct {
	A int
}

type streamDeclineEmbedProbe struct {
	Calls int `json:"-"`
	streamDeclineEmbedInner
	B int
}

func (p *streamDeclineEmbedProbe) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	p.Calls++
	return errors.ErrUnsupported
}

func TestFallbackFromUnexportedEmbedDecodes(t *testing.T) {
	data := `{"v":{"A":1,"B":2},"after":17}`
	var got streamTarget[streamDeclineEmbedProbe]
	s := NewScanner([]byte(data), (*streamDeclineEmbedProbe)(nil))
	s.Object()
	for s.Next() {
		switch s.Key() {
		case "v":
			if !QuotedFrom[streamDeclineEmbedProbe](&s, &got.V) {
				FallbackFrom(&s, &got.V)
			}
		case "after":
			Int(&s, &got.After)
		}
	}
	if err := s.End(); err != nil {
		t.Fatal(err)
	}
	if got.V.A != 1 || got.V.B != 2 || got.After != 17 {
		t.Fatalf("got %#v", got)
	}
}
