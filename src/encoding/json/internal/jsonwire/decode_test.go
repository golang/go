// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsonwire

import (
	"errors"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestConsumeWhitespace(t *testing.T) {
	tests := []struct {
		in   string
		want int
	}{
		{"", 0},
		{"a", 0},
		{" a", 1},
		{" a ", 1},
		{" \n\r\ta", 4},
		{" \n\r\t \n\r\t \n\r\t \n\r\t", 16},
		{"\u00a0", 0}, // non-breaking space is not JSON whitespace
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			if got := ConsumeWhitespace([]byte(tt.in)); got != tt.want {
				t.Errorf("ConsumeWhitespace(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestConsumeLiteral(t *testing.T) {
	tests := []struct {
		literal string
		in      string
		want    int
		wantErr error
	}{
		{"null", "", 0, io.ErrUnexpectedEOF},
		{"null", "n", 1, io.ErrUnexpectedEOF},
		{"null", "nu", 2, io.ErrUnexpectedEOF},
		{"null", "nul", 3, io.ErrUnexpectedEOF},
		{"null", "null", 4, nil},
		{"null", "nullx", 4, nil},
		{"null", "x", 0, NewInvalidCharacterError("x", "in literal null (expecting 'n')")},
		{"null", "nuxx", 2, NewInvalidCharacterError("x", "in literal null (expecting 'l')")},

		{"false", "", 0, io.ErrUnexpectedEOF},
		{"false", "f", 1, io.ErrUnexpectedEOF},
		{"false", "fa", 2, io.ErrUnexpectedEOF},
		{"false", "fal", 3, io.ErrUnexpectedEOF},
		{"false", "fals", 4, io.ErrUnexpectedEOF},
		{"false", "false", 5, nil},
		{"false", "falsex", 5, nil},
		{"false", "x", 0, NewInvalidCharacterError("x", "in literal false (expecting 'f')")},
		{"false", "falsx", 4, NewInvalidCharacterError("x", "in literal false (expecting 'e')")},

		{"true", "", 0, io.ErrUnexpectedEOF},
		{"true", "t", 1, io.ErrUnexpectedEOF},
		{"true", "tr", 2, io.ErrUnexpectedEOF},
		{"true", "tru", 3, io.ErrUnexpectedEOF},
		{"true", "true", 4, nil},
		{"true", "truex", 4, nil},
		{"true", "x", 0, NewInvalidCharacterError("x", "in literal true (expecting 't')")},
		{"true", "trux", 3, NewInvalidCharacterError("x", "in literal true (expecting 'e')")},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			var got int
			switch tt.literal {
			case "null":
				got = ConsumeNull([]byte(tt.in))
			case "false":
				got = ConsumeFalse([]byte(tt.in))
			case "true":
				got = ConsumeTrue([]byte(tt.in))
			default:
				t.Errorf("invalid literal: %v", tt.literal)
			}
			switch {
			case tt.wantErr == nil && got != tt.want:
				t.Errorf("Consume%v(%q) = %v, want %v", strings.Title(tt.literal), tt.in, got, tt.want)
			case tt.wantErr != nil && got != 0:
				t.Errorf("Consume%v(%q) = %v, want %v", strings.Title(tt.literal), tt.in, got, 0)
			}

			got, gotErr := ConsumeLiteral([]byte(tt.in), tt.literal)
			if got != tt.want || !reflect.DeepEqual(gotErr, tt.wantErr) {
				t.Errorf("ConsumeLiteral(%q, %q) = (%v, %v), want (%v, %v)", tt.in, tt.literal, got, gotErr, tt.want, tt.wantErr)
			}
		})
	}
}

func TestConsumeString(t *testing.T) {
	var errPrev = errors.New("same as previous error")
	tests := []struct {
		in             string
		simple         bool
		want           int
		wantUTF8       int // consumed bytes if validateUTF8 is specified
		wantFlags      ValueFlags
		wantUnquote    string
		wantErr        error
		wantErrUTF8    error // error if validateUTF8 is specified
		wantErrUnquote error
	}{
		{``, false, 0, 0, 0, "", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"`, false, 1, 1, 0, "", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`""`, true, 2, 2, 0, "", nil, nil, nil},
		{`""x`, true, 2, 2, 0, "", nil, nil, NewInvalidCharacterError("x", "after string value")},
		{` ""x`, false, 0, 0, 0, "", NewInvalidCharacterError(" ", "at start of string (expecting '\"')"), errPrev, errPrev},
		{`"hello`, false, 6, 6, 0, "hello", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"hello"`, true, 7, 7, 0, "hello", nil, nil, nil},
		{"\"\x00\"", false, 1, 1, stringNonVerbatim | stringNonCanonical, "", NewInvalidCharacterError("\x00", "in string (expecting non-control character)"), errPrev, errPrev},
		{`"\u0000"`, false, 8, 8, stringNonVerbatim, "\x00", nil, nil, nil},
		{"\"\x1f\"", false, 1, 1, stringNonVerbatim | stringNonCanonical, "", NewInvalidCharacterError("\x1f", "in string (expecting non-control character)"), errPrev, errPrev},
		{`"\u001f"`, false, 8, 8, stringNonVerbatim, "\x1f", nil, nil, nil},
		{`"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"`, true, 54, 54, 0, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", nil, nil, nil},
		{"\" !#$%'()*+,-./0123456789:;=?@[]^_`{|}~\x7f\"", true, 41, 41, 0, " !#$%'()*+,-./0123456789:;=?@[]^_`{|}~\x7f", nil, nil, nil},
		{`"&"`, false, 3, 3, 0, "&", nil, nil, nil},
		{`"<"`, false, 3, 3, 0, "<", nil, nil, nil},
		{`">"`, false, 3, 3, 0, ">", nil, nil, nil},
		{"\"x\x80\"", false, 4, 2, stringNonVerbatim | stringNonCanonical, "x\ufffd", nil, ErrInvalidUTF8, errPrev},
		{"\"x\xff\"", false, 4, 2, stringNonVerbatim | stringNonCanonical, "x\ufffd", nil, ErrInvalidUTF8, errPrev},
		{"\"x\xc0", false, 3, 2, stringNonVerbatim | stringNonCanonical, "x\ufffd", io.ErrUnexpectedEOF, ErrInvalidUTF8, io.ErrUnexpectedEOF},
		{"\"x\xc0\x80\"", false, 5, 2, stringNonVerbatim | stringNonCanonical, "x\ufffd\ufffd", nil, ErrInvalidUTF8, errPrev},
		{"\"x\xe0", false, 2, 2, 0, "x", io.ErrUnexpectedEOF, errPrev, errPrev},
		{"\"x\xe0\x80", false, 4, 2, stringNonVerbatim | stringNonCanonical, "x\ufffd\ufffd", io.ErrUnexpectedEOF, ErrInvalidUTF8, io.ErrUnexpectedEOF},
		{"\"x\xe0\x80\x80\"", false, 6, 2, stringNonVerbatim | stringNonCanonical, "x\ufffd\ufffd\ufffd", nil, ErrInvalidUTF8, errPrev},
		{"\"x\xf0", false, 2, 2, 0, "x", io.ErrUnexpectedEOF, errPrev, errPrev},
		{"\"x\xf0\x80", false, 4, 2, stringNonVerbatim | stringNonCanonical, "x\ufffd\ufffd", io.ErrUnexpectedEOF, ErrInvalidUTF8, io.ErrUnexpectedEOF},
		{"\"x\xf0\x80\x80", false, 5, 2, stringNonVerbatim | stringNonCanonical, "x\ufffd\ufffd\ufffd", io.ErrUnexpectedEOF, ErrInvalidUTF8, io.ErrUnexpectedEOF},
		{"\"x\xf0\x80\x80\x80\"", false, 7, 2, stringNonVerbatim | stringNonCanonical, "x\ufffd\ufffd\ufffd\ufffd", nil, ErrInvalidUTF8, errPrev},
		{"\"x\xed\xba\xad\"", false, 6, 2, stringNonVerbatim | stringNonCanonical, "x\ufffd\ufffd\ufffd", nil, ErrInvalidUTF8, errPrev},
		{"\"\u0080\u00f6\u20ac\ud799\ue000\ufb33\ufffd\U0001f602\"", false, 25, 25, 0, "\u0080\u00f6\u20ac\ud799\ue000\ufb33\ufffd\U0001f602", nil, nil, nil},
		{`"¢"`[:2], false, 1, 1, 0, "", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"¢"`[:3], false, 3, 3, 0, "¢", io.ErrUnexpectedEOF, errPrev, errPrev}, // missing terminating quote
		{`"¢"`[:4], false, 4, 4, 0, "¢", nil, nil, nil},
		{`"€"`[:2], false, 1, 1, 0, "", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"€"`[:3], false, 1, 1, 0, "", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"€"`[:4], false, 4, 4, 0, "€", io.ErrUnexpectedEOF, errPrev, errPrev}, // missing terminating quote
		{`"€"`[:5], false, 5, 5, 0, "€", nil, nil, nil},
		{`"𐍈"`[:2], false, 1, 1, 0, "", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"𐍈"`[:3], false, 1, 1, 0, "", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"𐍈"`[:4], false, 1, 1, 0, "", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"𐍈"`[:5], false, 5, 5, 0, "𐍈", io.ErrUnexpectedEOF, errPrev, errPrev}, // missing terminating quote
		{`"𐍈"`[:6], false, 6, 6, 0, "𐍈", nil, nil, nil},
		{`"x\`, false, 2, 2, stringNonVerbatim, "x", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"x\"`, false, 4, 4, stringNonVerbatim, "x\"", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"x\x"`, false, 2, 2, stringNonVerbatim | stringNonCanonical, "x", NewInvalidEscapeSequenceError(`\x`), errPrev, errPrev},
		{`"\"\\\b\f\n\r\t"`, false, 16, 16, stringNonVerbatim, "\"\\\b\f\n\r\t", nil, nil, nil},
		{`"/"`, true, 3, 3, 0, "/", nil, nil, nil},
		{`"\/"`, false, 4, 4, stringNonVerbatim | stringNonCanonical, "/", nil, nil, nil},
		{`"\u002f"`, false, 8, 8, stringNonVerbatim | stringNonCanonical, "/", nil, nil, nil},
		{`"\u`, false, 1, 1, stringNonVerbatim, "", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"\uf`, false, 1, 1, stringNonVerbatim, "", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"\uff`, false, 1, 1, stringNonVerbatim, "", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"\ufff`, false, 1, 1, stringNonVerbatim, "", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"\ufffd`, false, 7, 7, stringNonVerbatim | stringNonCanonical, "\ufffd", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"\ufffd"`, false, 8, 8, stringNonVerbatim | stringNonCanonical, "\ufffd", nil, nil, nil},
		{`"\uABCD"`, false, 8, 8, stringNonVerbatim | stringNonCanonical, "\uabcd", nil, nil, nil},
		{`"\uefX0"`, false, 1, 1, stringNonVerbatim | stringNonCanonical, "", NewInvalidEscapeSequenceError(`\uefX0`), errPrev, errPrev},
		{`"\uDEAD`, false, 7, 1, stringNonVerbatim | stringNonCanonical, "\ufffd", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"\uDEAD"`, false, 8, 1, stringNonVerbatim | stringNonCanonical, "\ufffd", nil, NewInvalidEscapeSequenceError(`\uDEAD"`), errPrev},
		{`"\uDEAD______"`, false, 14, 1, stringNonVerbatim | stringNonCanonical, "\ufffd______", nil, NewInvalidEscapeSequenceError(`\uDEAD______`), errPrev},
		{`"\uDEAD\uXXXX"`, false, 7, 1, stringNonVerbatim | stringNonCanonical, "\ufffd", NewInvalidEscapeSequenceError(`\uXXXX`), NewInvalidEscapeSequenceError(`\uDEAD\uXXXX`), NewInvalidEscapeSequenceError(`\uXXXX`)},
		{`"\uDEAD\uBEEF"`, false, 14, 1, stringNonVerbatim | stringNonCanonical, "\ufffd\ubeef", nil, NewInvalidEscapeSequenceError(`\uDEAD\uBEEF`), errPrev},
		{`"\uD800\udea`, false, 7, 1, stringNonVerbatim | stringNonCanonical, "\ufffd", io.ErrUnexpectedEOF, errPrev, errPrev},
		{`"\uD800\udb`, false, 7, 1, stringNonVerbatim | stringNonCanonical, "\ufffd", io.ErrUnexpectedEOF, NewInvalidEscapeSequenceError(`\uD800\udb`), io.ErrUnexpectedEOF},
		{`"\uD800\udead"`, false, 14, 14, stringNonVerbatim | stringNonCanonical, "\U000102ad", nil, nil, nil},
		{`"\u0022\u005c\u002f\u0008\u000c\u000a\u000d\u0009"`, false, 50, 50, stringNonVerbatim | stringNonCanonical, "\"\\/\b\f\n\r\t", nil, nil, nil},
		{`"\u0080\u00f6\u20ac\ud799\ue000\ufb33\ufffd\ud83d\ude02"`, false, 56, 56, stringNonVerbatim | stringNonCanonical, "\u0080\u00f6\u20ac\ud799\ue000\ufb33\ufffd\U0001f602", nil, nil, nil},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			if tt.wantErrUTF8 == errPrev {
				tt.wantErrUTF8 = tt.wantErr
			}
			if tt.wantErrUnquote == errPrev {
				tt.wantErrUnquote = tt.wantErrUTF8
			}

			switch got := ConsumeSimpleString([]byte(tt.in)); {
			case tt.simple && got != tt.want:
				t.Errorf("consumeSimpleString(%q) = %v, want %v", tt.in, got, tt.want)
			case !tt.simple && got != 0:
				t.Errorf("consumeSimpleString(%q) = %v, want %v", tt.in, got, 0)
			}

			var gotFlags ValueFlags
			got, gotErr := ConsumeString(&gotFlags, []byte(tt.in), false)
			if gotFlags != tt.wantFlags {
				t.Errorf("consumeString(%q, false) flags = %v, want %v", tt.in, gotFlags, tt.wantFlags)
			}
			if got != tt.want || !reflect.DeepEqual(gotErr, tt.wantErr) {
				t.Errorf("consumeString(%q, false) = (%v, %v), want (%v, %v)", tt.in, got, gotErr, tt.want, tt.wantErr)
			}

			got, gotErr = ConsumeString(&gotFlags, []byte(tt.in), true)
			if got != tt.wantUTF8 || !reflect.DeepEqual(gotErr, tt.wantErrUTF8) {
				t.Errorf("consumeString(%q, false) = (%v, %v), want (%v, %v)", tt.in, got, gotErr, tt.wantUTF8, tt.wantErrUTF8)
			}

			gotUnquote, gotErr := AppendUnquote(nil, tt.in)
			if string(gotUnquote) != tt.wantUnquote || !reflect.DeepEqual(gotErr, tt.wantErrUnquote) {
				t.Errorf("AppendUnquote(nil, %q) = (%q, %v), want (%q, %v)", tt.in[:got], gotUnquote, gotErr, tt.wantUnquote, tt.wantErrUnquote)
			}
		})
	}
}

func TestConsumeNumber(t *testing.T) {
	tests := []struct {
		in      string
		simple  bool
		want    int
		wantErr error
	}{
		{"", false, 0, io.ErrUnexpectedEOF},
		{`"NaN"`, false, 0, NewInvalidCharacterError("\"", "in number (expecting digit)")},
		{`"Infinity"`, false, 0, NewInvalidCharacterError("\"", "in number (expecting digit)")},
		{`"-Infinity"`, false, 0, NewInvalidCharacterError("\"", "in number (expecting digit)")},
		{".0", false, 0, NewInvalidCharacterError(".", "in number (expecting digit)")},
		{"0", true, 1, nil},
		{"-0", false, 2, nil},
		{"+0", false, 0, NewInvalidCharacterError("+", "in number (expecting digit)")},
		{"1", true, 1, nil},
		{"-1", false, 2, nil},
		{"00", true, 1, nil},
		{"-00", false, 2, nil},
		{"01", true, 1, nil},
		{"-01", false, 2, nil},
		{"0i", true, 1, nil},
		{"-0i", false, 2, nil},
		{"0f", true, 1, nil},
		{"-0f", false, 2, nil},
		{"9876543210", true, 10, nil},
		{"-9876543210", false, 11, nil},
		{"9876543210x", true, 10, nil},
		{"-9876543210x", false, 11, nil},
		{" 9876543210", true, 0, NewInvalidCharacterError(" ", "in number (expecting digit)")},
		{"- 9876543210", false, 1, NewInvalidCharacterError(" ", "in number (expecting digit)")},
		{strings.Repeat("9876543210", 1000), true, 10000, nil},
		{"-" + strings.Repeat("9876543210", 1000), false, 1 + 10000, nil},
		{"0.", false, 1, io.ErrUnexpectedEOF},
		{"-0.", false, 2, io.ErrUnexpectedEOF},
		{"0e", false, 1, io.ErrUnexpectedEOF},
		{"-0e", false, 2, io.ErrUnexpectedEOF},
		{"0E", false, 1, io.ErrUnexpectedEOF},
		{"-0E", false, 2, io.ErrUnexpectedEOF},
		{"0.0", false, 3, nil},
		{"-0.0", false, 4, nil},
		{"0e0", false, 3, nil},
		{"-0e0", false, 4, nil},
		{"0E0", false, 3, nil},
		{"-0E0", false, 4, nil},
		{"0.0123456789", false, 12, nil},
		{"-0.0123456789", false, 13, nil},
		{"1.f", false, 2, NewInvalidCharacterError("f", "in number (expecting digit)")},
		{"-1.f", false, 3, NewInvalidCharacterError("f", "in number (expecting digit)")},
		{"1.e", false, 2, NewInvalidCharacterError("e", "in number (expecting digit)")},
		{"-1.e", false, 3, NewInvalidCharacterError("e", "in number (expecting digit)")},
		{"1e0", false, 3, nil},
		{"-1e0", false, 4, nil},
		{"1E0", false, 3, nil},
		{"-1E0", false, 4, nil},
		{"1Ex", false, 2, NewInvalidCharacterError("x", "in number (expecting digit)")},
		{"-1Ex", false, 3, NewInvalidCharacterError("x", "in number (expecting digit)")},
		{"1e-0", false, 4, nil},
		{"-1e-0", false, 5, nil},
		{"1e+0", false, 4, nil},
		{"-1e+0", false, 5, nil},
		{"1E-0", false, 4, nil},
		{"-1E-0", false, 5, nil},
		{"1E+0", false, 4, nil},
		{"-1E+0", false, 5, nil},
		{"1E+00500", false, 8, nil},
		{"-1E+00500", false, 9, nil},
		{"1E+00500x", false, 8, nil},
		{"-1E+00500x", false, 9, nil},
		{"9876543210.0123456789e+01234589x", false, 31, nil},
		{"-9876543210.0123456789e+01234589x", false, 32, nil},
		{"1_000_000", true, 1, nil},
		{"0x12ef", true, 1, nil},
		{"0x1p-2", true, 1, nil},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			switch got := ConsumeSimpleNumber([]byte(tt.in)); {
			case tt.simple && got != tt.want:
				t.Errorf("ConsumeSimpleNumber(%q) = %v, want %v", tt.in, got, tt.want)
			case !tt.simple && got != 0:
				t.Errorf("ConsumeSimpleNumber(%q) = %v, want %v", tt.in, got, 0)
			}

			got, gotErr := ConsumeNumber([]byte(tt.in))
			if got != tt.want || !reflect.DeepEqual(gotErr, tt.wantErr) {
				t.Errorf("ConsumeNumber(%q) = (%v, %v), want (%v, %v)", tt.in, got, gotErr, tt.want, tt.wantErr)
			}
		})
	}
}

func TestParseHexUint16(t *testing.T) {
	tests := []struct {
		in     string
		want   uint16
		wantOk bool
	}{
		{"", 0, false},
		{"a", 0, false},
		{"ab", 0, false},
		{"abc", 0, false},
		{"abcd", 0xabcd, true},
		{"abcde", 0, false},
		{"9eA1", 0x9ea1, true},
		{"gggg", 0, false},
		{"0000", 0x0000, true},
		{"1234", 0x1234, true},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			got, gotOk := parseHexUint16([]byte(tt.in))
			if got != tt.want || gotOk != tt.wantOk {
				t.Errorf("parseHexUint16(%q) = (0x%04x, %v), want (0x%04x, %v)", tt.in, got, gotOk, tt.want, tt.wantOk)
			}
		})
	}
}

func TestParseUint(t *testing.T) {
	tests := []struct {
		in     string
		want   uint64
		wantOk bool
	}{
		{"", 0, false},
		{"0", 0, true},
		{"1", 1, true},
		{"-1", 0, false},
		{"1f", 0, false},
		{"00", 0, false},
		{"01", 0, false},
		{"10", 10, true},
		{"10.9", 0, false},
		{" 10", 0, false},
		{"10 ", 0, false},
		{"123456789", 123456789, true},
		{"123456789d", 0, false},
		{"18446744073709551614", math.MaxUint64 - 1, true},
		{"18446744073709551615", math.MaxUint64, true},
		{"18446744073709551616", math.MaxUint64, false},
		{"18446744073709551620", math.MaxUint64, false},
		{"18446744073709551700", math.MaxUint64, false},
		{"18446744073709552000", math.MaxUint64, false},
		{"18446744073709560000", math.MaxUint64, false},
		{"18446744073709600000", math.MaxUint64, false},
		{"18446744073710000000", math.MaxUint64, false},
		{"18446744073800000000", math.MaxUint64, false},
		{"18446744074000000000", math.MaxUint64, false},
		{"18446744080000000000", math.MaxUint64, false},
		{"18446744100000000000", math.MaxUint64, false},
		{"18446745000000000000", math.MaxUint64, false},
		{"18446750000000000000", math.MaxUint64, false},
		{"18446800000000000000", math.MaxUint64, false},
		{"18447000000000000000", math.MaxUint64, false},
		{"18450000000000000000", math.MaxUint64, false},
		{"18500000000000000000", math.MaxUint64, false},
		{"19000000000000000000", math.MaxUint64, false},
		{"19999999999999999999", math.MaxUint64, false},
		{"20000000000000000000", math.MaxUint64, false},
		{"100000000000000000000", math.MaxUint64, false},
		{"99999999999999999999999999999999", math.MaxUint64, false},
		{"99999999999999999999999999999999f", 0, false},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			got, gotOk := ParseUint([]byte(tt.in))
			if got != tt.want || gotOk != tt.wantOk {
				t.Errorf("ParseUint(%q) = (%v, %v), want (%v, %v)", tt.in, got, gotOk, tt.want, tt.wantOk)
			}
		})
	}
}

func TestParseFloat(t *testing.T) {
	tests := []struct {
		in     string
		want32 float64
		want64 float64
		wantOk bool
	}{
		{"0", 0, 0, true},
		{"-1", -1, -1, true},
		{"1", 1, 1, true},

		{"-16777215", -16777215, -16777215, true}, // -(1<<24 - 1)
		{"16777215", 16777215, 16777215, true},    // +(1<<24 - 1)
		{"-16777216", -16777216, -16777216, true}, // -(1<<24)
		{"16777216", 16777216, 16777216, true},    // +(1<<24)
		{"-16777217", -16777216, -16777217, true}, // -(1<<24 + 1)
		{"16777217", 16777216, 16777217, true},    // +(1<<24 + 1)

		{"-9007199254740991", -9007199254740992, -9007199254740991, true}, // -(1<<53 - 1)
		{"9007199254740991", 9007199254740992, 9007199254740991, true},    // +(1<<53 - 1)
		{"-9007199254740992", -9007199254740992, -9007199254740992, true}, // -(1<<53)
		{"9007199254740992", 9007199254740992, 9007199254740992, true},    // +(1<<53)
		{"-9007199254740993", -9007199254740992, -9007199254740992, true}, // -(1<<53 + 1)
		{"9007199254740993", 9007199254740992, 9007199254740992, true},    // +(1<<53 + 1)

		{"-1e1000", -math.MaxFloat32, -math.MaxFloat64, false},
		{"1e1000", +math.MaxFloat32, +math.MaxFloat64, false},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			got32, gotOk32 := ParseFloat([]byte(tt.in), 32)
			if got32 != tt.want32 || gotOk32 != tt.wantOk {
				t.Errorf("ParseFloat(%q, 32) = (%v, %v), want (%v, %v)", tt.in, got32, gotOk32, tt.want32, tt.wantOk)
			}

			got64, gotOk64 := ParseFloat([]byte(tt.in), 64)
			if got64 != tt.want64 || gotOk64 != tt.wantOk {
				t.Errorf("ParseFloat(%q, 64) = (%v, %v), want (%v, %v)", tt.in, got64, gotOk64, tt.want64, tt.wantOk)
			}
		})
	}
}
