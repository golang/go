// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"bytes"
	"errors"
	"io"
	"math/rand"
	"slices"
	"testing"

	"encoding/json/internal/jsontest"
)

func FuzzCoder(f *testing.F) {
	// Add a number of inputs to the corpus including valid and invalid data.
	for _, td := range coderTestdata {
		f.Add(int64(0), []byte(td.in))
	}
	for _, td := range decoderErrorTestdata {
		f.Add(int64(0), []byte(td.in))
	}
	for _, td := range encoderErrorTestdata {
		f.Add(int64(0), []byte(td.wantOut))
	}
	for _, td := range jsontest.Data {
		f.Add(int64(0), td.Data())
	}

	f.Fuzz(func(t *testing.T, seed int64, b []byte) {
		var tokVals []tokOrVal
		rn := rand.NewSource(seed)

		// Read a sequence of tokens or values. Skip the test for any errors
		// since we expect this with randomly generated fuzz inputs.
		src := bytes.NewReader(b)
		dec := NewDecoder(src)
		for {
			if rn.Int63()%8 > 0 {
				tok, err := dec.ReadToken()
				if err != nil {
					if err == io.EOF {
						break
					}
					t.Skipf("Decoder.ReadToken error: %v", err)
				}
				tokVals = append(tokVals, tok.Clone())
			} else {
				val, err := dec.ReadValue()
				if err != nil {
					if expectError := dec.PeekKind() == '}' || dec.PeekKind() == ']'; expectError {
						if _, ok := errors.AsType[*SyntacticError](err); ok {
							continue
						}
					}
					if err == io.EOF {
						break
					}
					t.Skipf("Decoder.ReadValue error: %v", err)
				}
				tokVals = append(tokVals, append(zeroValue, val...))
			}
		}

		// Write a sequence of tokens or values. Fail the test for any errors
		// since the previous stage guarantees that the input is valid.
		dst := new(bytes.Buffer)
		enc := NewEncoder(dst)
		for _, tokVal := range tokVals {
			switch tokVal := tokVal.(type) {
			case Token:
				if err := enc.WriteToken(tokVal); err != nil {
					t.Fatalf("Encoder.WriteToken error: %v", err)
				}
			case Value:
				if err := enc.WriteValue(tokVal); err != nil {
					t.Fatalf("Encoder.WriteValue error: %v", err)
				}
			}
		}

		// Encoded output and original input must decode to the same thing.
		var got, want []Token
		for dec := NewDecoder(bytes.NewReader(b)); dec.PeekKind() > 0; {
			tok, err := dec.ReadToken()
			if err != nil {
				t.Fatalf("Decoder.ReadToken error: %v", err)
			}
			got = append(got, tok.Clone())
		}
		for dec := NewDecoder(dst); dec.PeekKind() > 0; {
			tok, err := dec.ReadToken()
			if err != nil {
				t.Fatalf("Decoder.ReadToken error: %v", err)
			}
			want = append(want, tok.Clone())
		}
		if !equalTokens(got, want) {
			t.Fatalf("mismatching output:\ngot  %v\nwant %v", got, want)
		}
	})
}

func FuzzResumableDecoder(f *testing.F) {
	for _, td := range resumableDecoderTestdata {
		f.Add(int64(0), []byte(td))
	}

	f.Fuzz(func(t *testing.T, seed int64, b []byte) {
		rn := rand.NewSource(seed)

		// Regardless of how many bytes the underlying io.Reader produces,
		// the provided tokens, values, and errors should always be identical.
		t.Run("ReadToken", func(t *testing.T) {
			decGot := NewDecoder(&FaultyBuffer{B: b, MaxBytes: 8, Rand: rn})
			decWant := NewDecoder(bytes.NewReader(b))
			gotTok, gotErr := decGot.ReadToken()
			wantTok, wantErr := decWant.ReadToken()
			if gotTok.String() != wantTok.String() || !equalError(gotErr, wantErr) {
				t.Errorf("Decoder.ReadToken = (%v, %v), want (%v, %v)", gotTok, gotErr, wantTok, wantErr)
			}
		})
		t.Run("ReadValue", func(t *testing.T) {
			decGot := NewDecoder(&FaultyBuffer{B: b, MaxBytes: 8, Rand: rn})
			decWant := NewDecoder(bytes.NewReader(b))
			gotVal, gotErr := decGot.ReadValue()
			wantVal, wantErr := decWant.ReadValue()
			if !slices.Equal(gotVal, wantVal) || !equalError(gotErr, wantErr) {
				t.Errorf("Decoder.ReadValue = (%s, %v), want (%s, %v)", gotVal, gotErr, wantVal, wantErr)
			}
		})
	})
}

func FuzzValueFormat(f *testing.F) {
	for _, td := range valueTestdata {
		f.Add(int64(0), []byte(td.in))
	}

	// isValid reports whether b is valid according to the specified options.
	isValid := func(b []byte, opts ...Options) bool {
		d := NewDecoder(bytes.NewReader(b), opts...)
		_, errVal := d.ReadValue()
		_, errEOF := d.ReadToken()
		return errVal == nil && errEOF == io.EOF
	}

	// stripWhitespace removes all JSON whitespace characters from the input.
	stripWhitespace := func(in []byte) (out []byte) {
		out = make([]byte, 0, len(in))
		for _, c := range in {
			switch c {
			case ' ', '\n', '\r', '\t':
			default:
				out = append(out, c)
			}
		}
		return out
	}

	allOptions := []Options{
		AllowDuplicateNames(true),
		AllowInvalidUTF8(true),
		EscapeForHTML(true),
		EscapeForJS(true),
		PreserveRawStrings(true),
		CanonicalizeRawInts(true),
		CanonicalizeRawFloats(true),
		ReorderRawObjects(true),
		SpaceAfterColon(true),
		SpaceAfterComma(true),
		Multiline(true),
		WithIndent("\t"),
		WithIndentPrefix("    "),
	}

	f.Fuzz(func(t *testing.T, seed int64, b []byte) {
		validRFC7159 := isValid(b, AllowInvalidUTF8(true), AllowDuplicateNames(true))
		validRFC8259 := isValid(b, AllowInvalidUTF8(false), AllowDuplicateNames(true))
		validRFC7493 := isValid(b, AllowInvalidUTF8(false), AllowDuplicateNames(false))
		switch {
		case !validRFC7159 && validRFC8259:
			t.Errorf("invalid input per RFC 7159 implies invalid per RFC 8259")
		case !validRFC8259 && validRFC7493:
			t.Errorf("invalid input per RFC 8259 implies invalid per RFC 7493")
		}

		gotValid := Value(b).IsValid()
		wantValid := validRFC7493
		if gotValid != wantValid {
			t.Errorf("Value.IsValid = %v, want %v", gotValid, wantValid)
		}

		gotCompacted := Value(string(b))
		gotCompactOk := gotCompacted.Compact() == nil
		wantCompactOk := validRFC7159
		if !bytes.Equal(stripWhitespace(gotCompacted), stripWhitespace(b)) {
			t.Errorf("stripWhitespace(Value.Compact) = %s, want %s", stripWhitespace(gotCompacted), stripWhitespace(b))
		}
		if gotCompactOk != wantCompactOk {
			t.Errorf("Value.Compact success mismatch: got %v, want %v", gotCompactOk, wantCompactOk)
		}

		gotIndented := Value(string(b))
		gotIndentOk := gotIndented.Indent() == nil
		wantIndentOk := validRFC7159
		if !bytes.Equal(stripWhitespace(gotIndented), stripWhitespace(b)) {
			t.Errorf("stripWhitespace(Value.Indent) = %s, want %s", stripWhitespace(gotIndented), stripWhitespace(b))
		}
		if gotIndentOk != wantIndentOk {
			t.Errorf("Value.Indent success mismatch: got %v, want %v", gotIndentOk, wantIndentOk)
		}

		gotCanonicalized := Value(string(b))
		gotCanonicalizeOk := gotCanonicalized.Canonicalize() == nil
		wantCanonicalizeOk := validRFC7493
		if gotCanonicalizeOk != wantCanonicalizeOk {
			t.Errorf("Value.Canonicalize success mismatch: got %v, want %v", gotCanonicalizeOk, wantCanonicalizeOk)
		}

		// Random options should not result in a panic.
		var opts []Options
		rn := rand.New(rand.NewSource(seed))
		for _, opt := range allOptions {
			if rn.Intn(len(allOptions)/4) == 0 {
				opts = append(opts, opt)
			}
		}
		v := Value(b)
		v.Format(opts...) // should not panic
	})
}
