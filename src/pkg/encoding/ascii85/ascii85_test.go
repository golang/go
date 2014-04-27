// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ascii85

import (
	"bytes"
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

type testpair struct {
	decoded, encoded string
}

var pairs = []testpair{
	// Encode returns 0 when len(src) is 0
	{
		"",
		"",
	},
	// Wikipedia example
	{
		"Man is distinguished, not only by his reason, but by this singular passion from " +
			"other animals, which is a lust of the mind, that by a perseverance of delight in " +
			"the continued and indefatigable generation of knowledge, exceeds the short " +
			"vehemence of any carnal pleasure.",
		"9jqo^BlbD-BleB1DJ+*+F(f,q/0JhKF<GL>Cj@.4Gp$d7F!,L7@<6@)/0JDEF<G%<+EV:2F!,\n" +
			"O<DJ+*.@<*K0@<6L(Df-\\0Ec5e;DffZ(EZee.Bl.9pF\"AGXBPCsi+DGm>@3BB/F*&OCAfu2/AKY\n" +
			"i(DIb:@FD,*)+C]U=@3BN#EcYf8ATD3s@q?d$AftVqCh[NqF<G:8+EV:.+Cf>-FD5W8ARlolDIa\n" +
			"l(DId<j@<?3r@:F%a+D58'ATD4$Bl@l3De:,-DJs`8ARoFb/0JMK@qB4^F!,R<AKZ&-DfTqBG%G\n" +
			">uD.RTpAKYo'+CT/5+Cei#DII?(E,9)oF*2M7/c\n",
	},
	// Special case when shortening !!!!! to z.
	{
		"\000\000\000\000",
		"z",
	},
}

var bigtest = pairs[len(pairs)-1]

func testEqual(t *testing.T, msg string, args ...interface{}) bool {
	if args[len(args)-2] != args[len(args)-1] {
		t.Errorf(msg, args...)
		return false
	}
	return true
}

func strip85(s string) string {
	t := make([]byte, len(s))
	w := 0
	for r := 0; r < len(s); r++ {
		c := s[r]
		if c > ' ' {
			t[w] = c
			w++
		}
	}
	return string(t[0:w])
}

func TestEncode(t *testing.T) {
	for _, p := range pairs {
		buf := make([]byte, MaxEncodedLen(len(p.decoded)))
		n := Encode(buf, []byte(p.decoded))
		buf = buf[0:n]
		testEqual(t, "Encode(%q) = %q, want %q", p.decoded, strip85(string(buf)), strip85(p.encoded))
	}
}

func TestEncoder(t *testing.T) {
	for _, p := range pairs {
		bb := &bytes.Buffer{}
		encoder := NewEncoder(bb)
		encoder.Write([]byte(p.decoded))
		encoder.Close()
		testEqual(t, "Encode(%q) = %q, want %q", p.decoded, strip85(bb.String()), strip85(p.encoded))
	}
}

func TestEncoderBuffering(t *testing.T) {
	input := []byte(bigtest.decoded)
	for bs := 1; bs <= 12; bs++ {
		bb := &bytes.Buffer{}
		encoder := NewEncoder(bb)
		for pos := 0; pos < len(input); pos += bs {
			end := pos + bs
			if end > len(input) {
				end = len(input)
			}
			n, err := encoder.Write(input[pos:end])
			testEqual(t, "Write(%q) gave error %v, want %v", input[pos:end], err, error(nil))
			testEqual(t, "Write(%q) gave length %v, want %v", input[pos:end], n, end-pos)
		}
		err := encoder.Close()
		testEqual(t, "Close gave error %v, want %v", err, error(nil))
		testEqual(t, "Encoding/%d of %q = %q, want %q", bs, bigtest.decoded, strip85(bb.String()), strip85(bigtest.encoded))
	}
}

func TestDecode(t *testing.T) {
	for _, p := range pairs {
		dbuf := make([]byte, 4*len(p.encoded))
		ndst, nsrc, err := Decode(dbuf, []byte(p.encoded), true)
		testEqual(t, "Decode(%q) = error %v, want %v", p.encoded, err, error(nil))
		testEqual(t, "Decode(%q) = nsrc %v, want %v", p.encoded, nsrc, len(p.encoded))
		testEqual(t, "Decode(%q) = ndst %v, want %v", p.encoded, ndst, len(p.decoded))
		testEqual(t, "Decode(%q) = %q, want %q", p.encoded, string(dbuf[0:ndst]), p.decoded)
	}
}

func TestDecoder(t *testing.T) {
	for _, p := range pairs {
		decoder := NewDecoder(strings.NewReader(p.encoded))
		dbuf, err := ioutil.ReadAll(decoder)
		if err != nil {
			t.Fatal("Read failed", err)
		}
		testEqual(t, "Read from %q = length %v, want %v", p.encoded, len(dbuf), len(p.decoded))
		testEqual(t, "Decoding of %q = %q, want %q", p.encoded, string(dbuf), p.decoded)
		if err != nil {
			testEqual(t, "Read from %q = %v, want %v", p.encoded, err, io.EOF)
		}
	}
}

func TestDecoderBuffering(t *testing.T) {
	for bs := 1; bs <= 12; bs++ {
		decoder := NewDecoder(strings.NewReader(bigtest.encoded))
		buf := make([]byte, len(bigtest.decoded)+12)
		var total int
		for total = 0; total < len(bigtest.decoded); {
			n, err := decoder.Read(buf[total : total+bs])
			testEqual(t, "Read from %q at pos %d = %d, %v, want _, %v", bigtest.encoded, total, n, err, error(nil))
			total += n
		}
		testEqual(t, "Decoding/%d of %q = %q, want %q", bs, bigtest.encoded, string(buf[0:total]), bigtest.decoded)
	}
}

func TestDecodeCorrupt(t *testing.T) {
	type corrupt struct {
		e string
		p int
	}
	examples := []corrupt{
		{"v", 0},
		{"!z!!!!!!!!!", 1},
	}

	for _, e := range examples {
		dbuf := make([]byte, 4*len(e.e))
		_, _, err := Decode(dbuf, []byte(e.e), true)
		switch err := err.(type) {
		case CorruptInputError:
			testEqual(t, "Corruption in %q at offset %v, want %v", e.e, int(err), e.p)
		default:
			t.Error("Decoder failed to detect corruption in", e)
		}
	}
}

func TestBig(t *testing.T) {
	n := 3*1000 + 1
	raw := make([]byte, n)
	const alpha = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	for i := 0; i < n; i++ {
		raw[i] = alpha[i%len(alpha)]
	}
	encoded := new(bytes.Buffer)
	w := NewEncoder(encoded)
	nn, err := w.Write(raw)
	if nn != n || err != nil {
		t.Fatalf("Encoder.Write(raw) = %d, %v want %d, nil", nn, err, n)
	}
	err = w.Close()
	if err != nil {
		t.Fatalf("Encoder.Close() = %v want nil", err)
	}
	decoded, err := ioutil.ReadAll(NewDecoder(encoded))
	if err != nil {
		t.Fatalf("io.ReadAll(NewDecoder(...)): %v", err)
	}

	if !bytes.Equal(raw, decoded) {
		var i int
		for i = 0; i < len(decoded) && i < len(raw); i++ {
			if decoded[i] != raw[i] {
				break
			}
		}
		t.Errorf("Decode(Encode(%d-byte string)) failed at offset %d", n, i)
	}
}

func TestDecoderInternalWhitespace(t *testing.T) {
	s := strings.Repeat(" ", 2048) + "z"
	decoded, err := ioutil.ReadAll(NewDecoder(strings.NewReader(s)))
	if err != nil {
		t.Errorf("Decode gave error %v", err)
	}
	if want := []byte("\000\000\000\000"); !bytes.Equal(want, decoded) {
		t.Errorf("Decode failed: got %v, want %v", decoded, want)
	}
}
