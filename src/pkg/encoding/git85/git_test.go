// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package git85

import (
	"bytes"
	"io"
	"io/ioutil"
	"testing"
)

type testpair struct {
	decoded, encoded string
}

func testEqual(t *testing.T, msg string, args ...interface{}) bool {
	if args[len(args)-2] != args[len(args)-1] {
		t.Errorf(msg, args...)
		return false
	}
	return true
}

func TestGitTable(t *testing.T) {
	var saw [256]bool
	for i, c := range encode {
		if decode[c] != uint8(i+1) {
			t.Errorf("decode['%c'] = %d, want %d", c, decode[c], i+1)
		}
		saw[c] = true
	}
	for i, b := range saw {
		if !b && decode[i] != 0 {
			t.Errorf("decode[%d] = %d, want 0", i, decode[i])
		}
	}
}

var gitPairs = []testpair{
	// Wikipedia example, adapted.
	{
		"Man is distinguished, not only by his reason, but by this singular passion from " +
			"other animals, which is a lust of the mind, that by a perseverance of delight in " +
			"the continued and indefatigable generation of knowledge, exceeds the short " +
			"vehemence of any carnal pleasure.",

		"zO<`^zX>%ZCX>)XGZfA9Ab7*B`EFf-gbRchTY<VDJc_3(Mb0BhMVRLV8EFfZabRc4R\n" +
			"zAarPHb0BkRZfA9DVR9gFVRLh7Z*CxFa&K)QZ**v7av))DX>DO_b1WctXlY|;AZc?T\n" +
			"zVIXXEb95kYW*~HEWgu;7Ze%PVbZB98AYyqSVIXj2a&u*NWpZI|V`U(3W*}r`Y-wj`\n" +
			"zbRcPNAarPDAY*TCbZKsNWn>^>Ze$>7Ze(R<VRUI{VPb4$AZKN6WpZJ3X>V>IZ)PBC\n" +
			"zZf|#NWn^b%EFfigV`XJzb0BnRWgv5CZ*p`Xc4cT~ZDnp_Wgu^6AYpEKAY);2ZeeU7\n" +
			"IaBO8^b9HiME&u=k\n",
	},
}

var gitBigtest = gitPairs[len(gitPairs)-1]

func TestEncode(t *testing.T) {
	for _, p := range gitPairs {
		buf := make([]byte, EncodedLen(len(p.decoded)))
		n := Encode(buf, []byte(p.decoded))
		if n != len(buf) {
			t.Errorf("EncodedLen does not agree with Encode")
		}
		buf = buf[0:n]
		testEqual(t, "Encode(%q) = %q, want %q", p.decoded, string(buf), p.encoded)
	}
}

func TestEncoder(t *testing.T) {
	for _, p := range gitPairs {
		bb := &bytes.Buffer{}
		encoder := NewEncoder(bb)
		encoder.Write([]byte(p.decoded))
		encoder.Close()
		testEqual(t, "Encode(%q) = %q, want %q", p.decoded, bb.String(), p.encoded)
	}
}

func TestEncoderBuffering(t *testing.T) {
	input := []byte(gitBigtest.decoded)
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
		testEqual(t, "Encoding/%d of %q = %q, want %q", bs, gitBigtest.decoded, bb.String(), gitBigtest.encoded)
	}
}

func TestDecode(t *testing.T) {
	for _, p := range gitPairs {
		dbuf := make([]byte, 4*len(p.encoded))
		ndst, err := Decode(dbuf, []byte(p.encoded))
		testEqual(t, "Decode(%q) = error %v, want %v", p.encoded, err, error(nil))
		testEqual(t, "Decode(%q) = ndst %v, want %v", p.encoded, ndst, len(p.decoded))
		testEqual(t, "Decode(%q) = %q, want %q", p.encoded, string(dbuf[0:ndst]), p.decoded)
	}
}

func TestDecoder(t *testing.T) {
	for _, p := range gitPairs {
		decoder := NewDecoder(bytes.NewBufferString(p.encoded))
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
		decoder := NewDecoder(bytes.NewBufferString(gitBigtest.encoded))
		buf := make([]byte, len(gitBigtest.decoded)+12)
		var total int
		for total = 0; total < len(gitBigtest.decoded); {
			n, err := decoder.Read(buf[total : total+bs])
			testEqual(t, "Read from %q at pos %d = %d, %v, want _, %v", gitBigtest.encoded, total, n, err, error(nil))
			total += n
		}
		testEqual(t, "Decoding/%d of %q = %q, want %q", bs, gitBigtest.encoded, string(buf[0:total]), gitBigtest.decoded)
	}
}

func TestDecodeCorrupt(t *testing.T) {
	type corrupt struct {
		e string
		p int
	}
	examples := []corrupt{
		{"v", 0},
		{"!z!!!!!!!!!", 0},
	}

	for _, e := range examples {
		dbuf := make([]byte, 2*len(e.e))
		_, err := Decode(dbuf, []byte(e.e))
		switch err := err.(type) {
		case CorruptInputError:
			testEqual(t, "Corruption in %q at offset %v, want %v", e.e, int(err), e.p)
		default:
			t.Error("Decoder failed to detect corruption in", e)
		}
	}
}

func TestGitBig(t *testing.T) {
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
		t.Fatalf("ioutil.ReadAll(NewDecoder(...)): %v", err)
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
