// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsonwire

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"math"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"encoding/json/internal/jsonflags"
)

func TestAppendQuote(t *testing.T) {
	tests := []struct {
		in          string
		flags       jsonflags.Bools
		want        string
		wantErr     error
		wantErrUTF8 error
	}{
		{"", 0, `""`, nil, nil},
		{"hello", 0, `"hello"`, nil, nil},
		{"\x00", 0, `"\u0000"`, nil, nil},
		{"\x1f", 0, `"\u001f"`, nil, nil},
		{"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", 0, `"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"`, nil, nil},
		{" !#$%&'()*+,-./0123456789:;<=>?@[]^_`{|}~\x7f", 0, "\" !#$%&'()*+,-./0123456789:;<=>?@[]^_`{|}~\x7f\"", nil, nil},
		{" !#$%&'()*+,-./0123456789:;<=>?@[]^_`{|}~\x7f", jsonflags.EscapeForHTML, "\" !#$%\\u0026'()*+,-./0123456789:;\\u003c=\\u003e?@[]^_`{|}~\x7f\"", nil, nil},
		{" !#$%&'()*+,-./0123456789:;<=>?@[]^_`{|}~\x7f", jsonflags.EscapeForJS, "\" !#$%&'()*+,-./0123456789:;<=>?@[]^_`{|}~\x7f\"", nil, nil},
		{"\u2027\u2028\u2029\u2030", 0, "\"\u2027\u2028\u2029\u2030\"", nil, nil},
		{"\u2027\u2028\u2029\u2030", jsonflags.EscapeForHTML, "\"\u2027\u2028\u2029\u2030\"", nil, nil},
		{"\u2027\u2028\u2029\u2030", jsonflags.EscapeForJS, "\"\u2027\\u2028\\u2029\u2030\"", nil, nil},
		{"x\x80\ufffd", 0, "\"x\ufffd\ufffd\"", nil, ErrInvalidUTF8},
		{"x\xff\ufffd", 0, "\"x\ufffd\ufffd\"", nil, ErrInvalidUTF8},
		{"x\xc0", 0, "\"x\ufffd\"", nil, ErrInvalidUTF8},
		{"x\xc0\x80", 0, "\"x\ufffd\ufffd\"", nil, ErrInvalidUTF8},
		{"x\xe0", 0, "\"x\ufffd\"", nil, ErrInvalidUTF8},
		{"x\xe0\x80", 0, "\"x\ufffd\ufffd\"", nil, ErrInvalidUTF8},
		{"x\xe0\x80\x80", 0, "\"x\ufffd\ufffd\ufffd\"", nil, ErrInvalidUTF8},
		{"x\xf0", 0, "\"x\ufffd\"", nil, ErrInvalidUTF8},
		{"x\xf0\x80", 0, "\"x\ufffd\ufffd\"", nil, ErrInvalidUTF8},
		{"x\xf0\x80\x80", 0, "\"x\ufffd\ufffd\ufffd\"", nil, ErrInvalidUTF8},
		{"x\xf0\x80\x80\x80", 0, "\"x\ufffd\ufffd\ufffd\ufffd\"", nil, ErrInvalidUTF8},
		{"x\xed\xba\xad", 0, "\"x\ufffd\ufffd\ufffd\"", nil, ErrInvalidUTF8},
		{"\"\\/\b\f\n\r\t", 0, `"\"\\/\b\f\n\r\t"`, nil, nil},
		{"٩(-̮̮̃-̃)۶ ٩(●̮̮̃•̃)۶ ٩(͡๏̯͡๏)۶ ٩(-̮̮̃•̃).", 0, `"٩(-̮̮̃-̃)۶ ٩(●̮̮̃•̃)۶ ٩(͡๏̯͡๏)۶ ٩(-̮̮̃•̃)."`, nil, nil},
		{"\u0080\u00f6\u20ac\ud799\ue000\ufb33\ufffd\U0001f602", 0, "\"\u0080\u00f6\u20ac\ud799\ue000\ufb33\ufffd\U0001f602\"", nil, nil},
		{"\u0000\u001f\u0020\u0022\u0026\u003c\u003e\u005c\u007f\u0080\u2028\u2029\ufffd\U0001f602", 0, "\"\\u0000\\u001f\u0020\\\"\u0026\u003c\u003e\\\\\u007f\u0080\u2028\u2029\ufffd\U0001f602\"", nil, nil},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			var flags jsonflags.Flags
			flags.Set(tt.flags | 1)

			flags.Set(jsonflags.AllowInvalidUTF8 | 1)
			got, gotErr := AppendQuote(nil, tt.in, &flags)
			if string(got) != tt.want || !reflect.DeepEqual(gotErr, tt.wantErr) {
				t.Errorf("AppendQuote(nil, %q, ...) = (%s, %v), want (%s, %v)", tt.in, got, gotErr, tt.want, tt.wantErr)
			}
			flags.Set(jsonflags.AllowInvalidUTF8 | 0)
			switch got, gotErr := AppendQuote(nil, tt.in, &flags); {
			case tt.wantErrUTF8 == nil && (string(got) != tt.want || !reflect.DeepEqual(gotErr, tt.wantErr)):
				t.Errorf("AppendQuote(nil, %q, ...) = (%s, %v), want (%s, %v)", tt.in, got, gotErr, tt.want, tt.wantErr)
			case tt.wantErrUTF8 != nil && (!strings.HasPrefix(tt.want, string(got)) || !reflect.DeepEqual(gotErr, tt.wantErrUTF8)):
				t.Errorf("AppendQuote(nil, %q, ...) = (%s, %v), want (%s, %v)", tt.in, got, gotErr, tt.want, tt.wantErrUTF8)
			}
		})
	}
}

func TestAppendNumber(t *testing.T) {
	tests := []struct {
		in     float64
		want32 string
		want64 string
	}{
		{math.E, "2.7182817", "2.718281828459045"},
		{math.Pi, "3.1415927", "3.141592653589793"},
		{math.SmallestNonzeroFloat32, "1e-45", "1.401298464324817e-45"},
		{math.SmallestNonzeroFloat64, "0", "5e-324"},
		{math.MaxFloat32, "3.4028235e+38", "3.4028234663852886e+38"},
		{math.MaxFloat64, "", "1.7976931348623157e+308"},
		{0.1111111111111111, "0.11111111", "0.1111111111111111"},
		{0.2222222222222222, "0.22222222", "0.2222222222222222"},
		{0.3333333333333333, "0.33333334", "0.3333333333333333"},
		{0.4444444444444444, "0.44444445", "0.4444444444444444"},
		{0.5555555555555555, "0.5555556", "0.5555555555555555"},
		{0.6666666666666666, "0.6666667", "0.6666666666666666"},
		{0.7777777777777777, "0.7777778", "0.7777777777777777"},
		{0.8888888888888888, "0.8888889", "0.8888888888888888"},
		{0.9999999999999999, "1", "0.9999999999999999"},

		// The following entries are from RFC 8785, appendix B
		// which are designed to ensure repeatable formatting of 64-bit floats.
		{math.Float64frombits(0x0000000000000000), "0", "0"},
		{math.Float64frombits(0x8000000000000000), "-0", "-0"}, // differs from RFC 8785
		{math.Float64frombits(0x0000000000000001), "0", "5e-324"},
		{math.Float64frombits(0x8000000000000001), "-0", "-5e-324"},
		{math.Float64frombits(0x7fefffffffffffff), "", "1.7976931348623157e+308"},
		{math.Float64frombits(0xffefffffffffffff), "", "-1.7976931348623157e+308"},
		{math.Float64frombits(0x4340000000000000), "9007199000000000", "9007199254740992"},
		{math.Float64frombits(0xc340000000000000), "-9007199000000000", "-9007199254740992"},
		{math.Float64frombits(0x4430000000000000), "295147900000000000000", "295147905179352830000"},
		{math.Float64frombits(0x44b52d02c7e14af5), "1e+23", "9.999999999999997e+22"},
		{math.Float64frombits(0x44b52d02c7e14af6), "1e+23", "1e+23"},
		{math.Float64frombits(0x44b52d02c7e14af7), "1e+23", "1.0000000000000001e+23"},
		{math.Float64frombits(0x444b1ae4d6e2ef4e), "1e+21", "999999999999999700000"},
		{math.Float64frombits(0x444b1ae4d6e2ef4f), "1e+21", "999999999999999900000"},
		{math.Float64frombits(0x444b1ae4d6e2ef50), "1e+21", "1e+21"},
		{math.Float64frombits(0x3eb0c6f7a0b5ed8c), "0.000001", "9.999999999999997e-7"},
		{math.Float64frombits(0x3eb0c6f7a0b5ed8d), "0.000001", "0.000001"},
		{math.Float64frombits(0x41b3de4355555553), "333333340", "333333333.3333332"},
		{math.Float64frombits(0x41b3de4355555554), "333333340", "333333333.33333325"},
		{math.Float64frombits(0x41b3de4355555555), "333333340", "333333333.3333333"},
		{math.Float64frombits(0x41b3de4355555556), "333333340", "333333333.3333334"},
		{math.Float64frombits(0x41b3de4355555557), "333333340", "333333333.33333343"},
		{math.Float64frombits(0xbecbf647612f3696), "-0.0000033333333", "-0.0000033333333333333333"},
		{math.Float64frombits(0x43143ff3c1cb0959), "1424953900000000", "1424953923781206.2"},

		// The following are select entries from RFC 8785, appendix B,
		// but modified for equivalent 32-bit behavior.
		{float64(math.Float32frombits(0x65a96815)), "9.999999e+22", "9.999998877476383e+22"},
		{float64(math.Float32frombits(0x65a96816)), "1e+23", "9.999999778196308e+22"},
		{float64(math.Float32frombits(0x65a96817)), "1.0000001e+23", "1.0000000678916234e+23"},
		{float64(math.Float32frombits(0x6258d725)), "999999900000000000000", "999999879303389000000"},
		{float64(math.Float32frombits(0x6258d726)), "999999950000000000000", "999999949672133200000"},
		{float64(math.Float32frombits(0x6258d727)), "1e+21", "1.0000000200408773e+21"},
		{float64(math.Float32frombits(0x6258d728)), "1.0000001e+21", "1.0000000904096215e+21"},
		{float64(math.Float32frombits(0x358637bc)), "9.999999e-7", "9.99999883788405e-7"},
		{float64(math.Float32frombits(0x358637bd)), "0.000001", "9.999999974752427e-7"},
		{float64(math.Float32frombits(0x358637be)), "0.0000010000001", "0.0000010000001111620804"},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			if got32 := string(AppendFloat(nil, tt.in, 32)); got32 != tt.want32 && tt.want32 != "" {
				t.Errorf("AppendFloat(nil, %v, 32) = %v, want %v", tt.in, got32, tt.want32)
			}
			if got64 := string(AppendFloat(nil, tt.in, 64)); got64 != tt.want64 && tt.want64 != "" {
				t.Errorf("AppendFloat(nil, %v, 64) = %v, want %v", tt.in, got64, tt.want64)
			}
		})
	}
}

// The default of 1e4 lines was chosen since it is sufficiently large to include
// test numbers from all three categories (i.e., static, series, and random).
// Yet, it is sufficiently low to execute quickly relative to other tests.
//
// Processing 1e8 lines takes a minute and processes about 4GiB worth of text.
var testCanonicalNumberLines = flag.Float64("canonical-number-lines", 1e4, "specify the number of lines to check from the canonical numbers testdata")

// TestCanonicalNumber verifies that appendNumber complies with RFC 8785
// according to the testdata provided by the reference implementation.
// See https://github.com/cyberphone/json-canonicalization/tree/master/testdata#es6-numbers.
func TestCanonicalNumber(t *testing.T) {
	const testfileURL = "https://github.com/cyberphone/json-canonicalization/releases/download/es6testfile/es6testfile100m.txt.gz"
	hashes := map[float64]string{
		1e3: "be18b62b6f69cdab33a7e0dae0d9cfa869fda80ddc712221570f9f40a5878687",
		1e4: "b9f7a8e75ef22a835685a52ccba7f7d6bdc99e34b010992cbc5864cd12be6892",
		1e5: "22776e6d4b49fa294a0d0f349268e5c28808fe7e0cb2bcbe28f63894e494d4c7",
		1e6: "49415fee2c56c77864931bd3624faad425c3c577d6d74e89a83bc725506dad16",
		1e7: "b9f8a44a91d46813b21b9602e72f112613c91408db0b8341fb94603d9db135e0",
		1e8: "0f7dda6b0837dde083c5d6b896f7d62340c8a2415b0c7121d83145e08a755272",
	}
	wantHash := hashes[*testCanonicalNumberLines]
	if wantHash == "" {
		t.Fatalf("canonical-number-lines must be one of the following values: 1e3, 1e4, 1e5, 1e6, 1e7, 1e8")
	}
	numLines := int(*testCanonicalNumberLines)

	// generator returns a function that generates the next float64 to format.
	// This implements the algorithm specified in the reference implementation.
	generator := func() func() float64 {
		static := [...]uint64{
			0x0000000000000000, 0x8000000000000000, 0x0000000000000001, 0x8000000000000001,
			0xc46696695dbd1cc3, 0xc43211ede4974a35, 0xc3fce97ca0f21056, 0xc3c7213080c1a6ac,
			0xc39280f39a348556, 0xc35d9b1f5d20d557, 0xc327af4c4a80aaac, 0xc2f2f2a36ecd5556,
			0xc2be51057e155558, 0xc28840d131aaaaac, 0xc253670dc1555557, 0xc21f0b4935555557,
			0xc1e8d5d42aaaaaac, 0xc1b3de4355555556, 0xc17fca0555555556, 0xc1496e6aaaaaaaab,
			0xc114585555555555, 0xc0e046aaaaaaaaab, 0xc0aa0aaaaaaaaaaa, 0xc074d55555555555,
			0xc040aaaaaaaaaaab, 0xc00aaaaaaaaaaaab, 0xbfd5555555555555, 0xbfa1111111111111,
			0xbf6b4e81b4e81b4f, 0xbf35d867c3ece2a5, 0xbf0179ec9cbd821e, 0xbecbf647612f3696,
			0xbe965e9f80f29212, 0xbe61e54c672874db, 0xbe2ca213d840baf8, 0xbdf6e80fe033c8c6,
			0xbdc2533fe68fd3d2, 0xbd8d51ffd74c861c, 0xbd5774ccac3d3817, 0xbd22c3d6f030f9ac,
			0xbcee0624b3818f79, 0xbcb804ea293472c7, 0xbc833721ba905bd3, 0xbc4ebe9c5db3c61e,
			0xbc18987d17c304e5, 0xbbe3ad30dfcf371d, 0xbbaf7b816618582f, 0xbb792f9ab81379bf,
			0xbb442615600f9499, 0xbb101e77800c76e1, 0xbad9ca58cce0be35, 0xbaa4a1e0a3e6fe90,
			0xba708180831f320d, 0xba3a68cd9e985016, 0x446696695dbd1cc3, 0x443211ede4974a35,
			0x43fce97ca0f21056, 0x43c7213080c1a6ac, 0x439280f39a348556, 0x435d9b1f5d20d557,
			0x4327af4c4a80aaac, 0x42f2f2a36ecd5556, 0x42be51057e155558, 0x428840d131aaaaac,
			0x4253670dc1555557, 0x421f0b4935555557, 0x41e8d5d42aaaaaac, 0x41b3de4355555556,
			0x417fca0555555556, 0x41496e6aaaaaaaab, 0x4114585555555555, 0x40e046aaaaaaaaab,
			0x40aa0aaaaaaaaaaa, 0x4074d55555555555, 0x4040aaaaaaaaaaab, 0x400aaaaaaaaaaaab,
			0x3fd5555555555555, 0x3fa1111111111111, 0x3f6b4e81b4e81b4f, 0x3f35d867c3ece2a5,
			0x3f0179ec9cbd821e, 0x3ecbf647612f3696, 0x3e965e9f80f29212, 0x3e61e54c672874db,
			0x3e2ca213d840baf8, 0x3df6e80fe033c8c6, 0x3dc2533fe68fd3d2, 0x3d8d51ffd74c861c,
			0x3d5774ccac3d3817, 0x3d22c3d6f030f9ac, 0x3cee0624b3818f79, 0x3cb804ea293472c7,
			0x3c833721ba905bd3, 0x3c4ebe9c5db3c61e, 0x3c18987d17c304e5, 0x3be3ad30dfcf371d,
			0x3baf7b816618582f, 0x3b792f9ab81379bf, 0x3b442615600f9499, 0x3b101e77800c76e1,
			0x3ad9ca58cce0be35, 0x3aa4a1e0a3e6fe90, 0x3a708180831f320d, 0x3a3a68cd9e985016,
			0x4024000000000000, 0x4014000000000000, 0x3fe0000000000000, 0x3fa999999999999a,
			0x3f747ae147ae147b, 0x3f40624dd2f1a9fc, 0x3f0a36e2eb1c432d, 0x3ed4f8b588e368f1,
			0x3ea0c6f7a0b5ed8d, 0x3e6ad7f29abcaf48, 0x3e35798ee2308c3a, 0x3ed539223589fa95,
			0x3ed4ff26cd5a7781, 0x3ed4f95a762283ff, 0x3ed4f8c60703520c, 0x3ed4f8b72f19cd0d,
			0x3ed4f8b5b31c0c8d, 0x3ed4f8b58d1c461a, 0x3ed4f8b5894f7f0e, 0x3ed4f8b588ee37f3,
			0x3ed4f8b588e47da4, 0x3ed4f8b588e3849c, 0x3ed4f8b588e36bb5, 0x3ed4f8b588e36937,
			0x3ed4f8b588e368f8, 0x3ed4f8b588e368f1, 0x3ff0000000000000, 0xbff0000000000000,
			0xbfeffffffffffffa, 0xbfeffffffffffffb, 0x3feffffffffffffa, 0x3feffffffffffffb,
			0x3feffffffffffffc, 0x3feffffffffffffe, 0xbfefffffffffffff, 0xbfefffffffffffff,
			0x3fefffffffffffff, 0x3fefffffffffffff, 0x3fd3333333333332, 0x3fd3333333333333,
			0x3fd3333333333334, 0x0010000000000000, 0x000ffffffffffffd, 0x000fffffffffffff,
			0x7fefffffffffffff, 0xffefffffffffffff, 0x4340000000000000, 0xc340000000000000,
			0x4430000000000000, 0x44b52d02c7e14af5, 0x44b52d02c7e14af6, 0x44b52d02c7e14af7,
			0x444b1ae4d6e2ef4e, 0x444b1ae4d6e2ef4f, 0x444b1ae4d6e2ef50, 0x3eb0c6f7a0b5ed8c,
			0x3eb0c6f7a0b5ed8d, 0x41b3de4355555553, 0x41b3de4355555554, 0x41b3de4355555555,
			0x41b3de4355555556, 0x41b3de4355555557, 0xbecbf647612f3696, 0x43143ff3c1cb0959,
		}
		var state struct {
			idx   int
			data  []byte
			block [sha256.Size]byte
		}
		return func() float64 {
			const numSerial = 2000
			var f float64
			switch {
			case state.idx < len(static):
				f = math.Float64frombits(static[state.idx])
			case state.idx < len(static)+numSerial:
				f = math.Float64frombits(0x0010000000000000 + uint64(state.idx-len(static)))
			default:
				for f == 0 || math.IsNaN(f) || math.IsInf(f, 0) {
					if len(state.data) == 0 {
						state.block = sha256.Sum256(state.block[:])
						state.data = state.block[:]
					}
					f = math.Float64frombits(binary.LittleEndian.Uint64(state.data))
					state.data = state.data[8:]
				}
			}
			state.idx++
			return f
		}
	}

	// Pass through the test twice. In the first pass we only hash the output,
	// while in the second pass we check every line against the golden testdata.
	// If the hashes match in the first pass, then we skip the second pass.
	for _, checkGolden := range []bool{false, true} {
		var br *bufio.Reader // for line-by-line reading of es6testfile100m.txt
		if checkGolden {
			resp, err := http.Get(testfileURL)
			if err != nil {
				t.Fatalf("http.Get error: %v", err)
			}
			defer resp.Body.Close()

			zr, err := gzip.NewReader(resp.Body)
			if err != nil {
				t.Fatalf("gzip.NewReader error: %v", err)
			}

			br = bufio.NewReader(zr)
		}

		// appendNumberJCS differs from appendNumber only for -0.
		appendNumberJCS := func(b []byte, f float64) []byte {
			if math.Signbit(f) && f == 0 {
				return append(b, '0')
			}
			return AppendFloat(b, f, 64)
		}

		var gotLine []byte
		next := generator()
		hash := sha256.New()
		start := time.Now()
		lastPrint := start
		for n := 1; n <= numLines; n++ {
			// Generate the formatted line for this number.
			f := next()
			gotLine = gotLine[:0] // reset from previous usage
			gotLine = strconv.AppendUint(gotLine, math.Float64bits(f), 16)
			gotLine = append(gotLine, ',')
			gotLine = appendNumberJCS(gotLine, f)
			gotLine = append(gotLine, '\n')
			hash.Write(gotLine)

			// Check that the formatted line matches.
			if checkGolden {
				wantLine, err := br.ReadBytes('\n')
				if err != nil {
					t.Fatalf("bufio.Reader.ReadBytes error: %v", err)
				}
				if !bytes.Equal(gotLine, wantLine) {
					t.Errorf("mismatch on line %d:\n\tgot  %v\n\twant %v",
						n, strings.TrimSpace(string(gotLine)), strings.TrimSpace(string(wantLine)))
				}
			}

			// Print progress.
			if now := time.Now(); now.Sub(lastPrint) > time.Second || n == numLines {
				remaining := float64(now.Sub(start)) * float64(numLines-n) / float64(n)
				t.Logf("%0.3f%% (%v remaining)",
					100.0*float64(n)/float64(numLines),
					time.Duration(remaining).Round(time.Second))
				lastPrint = now
			}
		}

		gotHash := hex.EncodeToString(hash.Sum(nil))
		if gotHash == wantHash {
			return // hashes match, no need to check golden testdata
		}
	}
}
