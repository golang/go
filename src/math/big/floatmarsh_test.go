// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
)

var floatVals = []string{
	"0",
	"1",
	"0.1",
	"2.71828",
	"1234567890",
	"3.14e1234",
	"3.14e-1234",
	"0.738957395793475734757349579759957975985497e100",
	"0.73895739579347546656564656573475734957975995797598589749859834759476745986795497e100",
	"inf",
	"Inf",
}

func TestFloatGobEncoding(t *testing.T) {
	var medium bytes.Buffer
	enc := gob.NewEncoder(&medium)
	dec := gob.NewDecoder(&medium)
	for _, test := range floatVals {
		for _, sign := range []string{"", "+", "-"} {
			for _, prec := range []uint{0, 1, 2, 10, 53, 64, 100, 1000} {
				for _, mode := range []RoundingMode{ToNearestEven, ToNearestAway, ToZero, AwayFromZero, ToNegativeInf, ToPositiveInf} {
					medium.Reset() // empty buffer for each test case (in case of failures)
					x := sign + test

					var tx Float
					_, _, err := tx.SetPrec(prec).SetMode(mode).Parse(x, 0)
					if err != nil {
						t.Errorf("parsing of %s (%dbits, %v) failed (invalid test case): %v", x, prec, mode, err)
						continue
					}

					// If tx was set to prec == 0, tx.Parse(x, 0) assumes precision 64. Correct it.
					if prec == 0 {
						tx.SetPrec(0)
					}

					if err := enc.Encode(&tx); err != nil {
						t.Errorf("encoding of %v (%dbits, %v) failed: %v", &tx, prec, mode, err)
						continue
					}

					var rx Float
					if err := dec.Decode(&rx); err != nil {
						t.Errorf("decoding of %v (%dbits, %v) failed: %v", &tx, prec, mode, err)
						continue
					}

					if rx.Cmp(&tx) != 0 {
						t.Errorf("transmission of %s failed: got %s want %s", x, rx.String(), tx.String())
						continue
					}

					if rx.Prec() != prec {
						t.Errorf("transmission of %s's prec failed: got %d want %d", x, rx.Prec(), prec)
					}

					if rx.Mode() != mode {
						t.Errorf("transmission of %s's mode failed: got %s want %s", x, rx.Mode(), mode)
					}

					if rx.Acc() != tx.Acc() {
						t.Errorf("transmission of %s's accuracy failed: got %s want %s", x, rx.Acc(), tx.Acc())
					}
				}
			}
		}
	}
}

func TestFloatCorruptGob(t *testing.T) {
	var buf bytes.Buffer
	tx := NewFloat(4 / 3).SetPrec(1000).SetMode(ToPositiveInf)
	if err := gob.NewEncoder(&buf).Encode(tx); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()

	var rx Float
	if err := gob.NewDecoder(bytes.NewReader(b)).Decode(&rx); err != nil {
		t.Fatal(err)
	}

	if err := gob.NewDecoder(bytes.NewReader(b[:10])).Decode(&rx); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("got %v want EOF", err)
	}

	b[1] = 0
	if err := gob.NewDecoder(bytes.NewReader(b)).Decode(&rx); err == nil {
		t.Fatal("got nil want version error")
	}
}

func TestFloatJSONEncoding(t *testing.T) {
	for _, test := range floatVals {
		for _, sign := range []string{"", "+", "-"} {
			for _, prec := range []uint{0, 1, 2, 10, 53, 64, 100, 1000} {
				if prec > 53 && testing.Short() {
					continue
				}
				x := sign + test
				var tx Float
				_, _, err := tx.SetPrec(prec).Parse(x, 0)
				if err != nil {
					t.Errorf("parsing of %s (prec = %d) failed (invalid test case): %v", x, prec, err)
					continue
				}
				b, err := json.Marshal(&tx)
				if err != nil {
					t.Errorf("marshaling of %v (prec = %d) failed: %v", &tx, prec, err)
					continue
				}
				var rx Float
				rx.SetPrec(prec)
				if err := json.Unmarshal(b, &rx); err != nil {
					t.Errorf("unmarshaling of %v (prec = %d) failed: %v", &tx, prec, err)
					continue
				}
				if rx.Cmp(&tx) != 0 {
					t.Errorf("JSON encoding of %v (prec = %d) failed: got %v want %v", &tx, prec, &rx, &tx)
				}
			}
		}
	}
}

func TestFloatGobDecodeShortBuffer(t *testing.T) {
	for _, tc := range [][]byte{
		[]byte{0x1, 0x0, 0x0, 0x0},
		[]byte{0x1, 0xfa, 0x0, 0x0, 0x0, 0x0},
	} {
		err := NewFloat(0).GobDecode(tc)
		if err == nil {
			t.Error("expected GobDecode to return error for malformed input")
		}
	}
}

func TestFloatGobDecodeInvalid(t *testing.T) {
	for _, tc := range []struct {
		buf []byte
		msg string
	}{
		{
			[]byte{0x1, 0x2a, 0x20, 0x20, 0x20, 0x20, 0x0, 0x20, 0x20, 0x20, 0x0, 0x20, 0x20, 0x20, 0x20, 0x0, 0x0, 0x0, 0x0, 0xc},
			"Float.GobDecode: msb not set in last word",
		},
		{
			[]byte{1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			"Float.GobDecode: nonzero finite number with empty mantissa",
		},
	} {
		err := NewFloat(0).GobDecode(tc.buf)
		if err == nil || !strings.HasPrefix(err.Error(), tc.msg) {
			t.Errorf("expected GobDecode error prefix: %s, got: %v", tc.msg, err)
		}
	}
}
