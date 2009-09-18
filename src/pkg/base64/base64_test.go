// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base64

import (
	"bytes";
	"io";
	"os";
	"reflect";
	"strings";
	"testing";
)

type testpair struct {
	decoded, encoded string;
}

var pairs = []testpair {
	// RFC 3548 examples
	testpair{"\x14\xfb\x9c\x03\xd9\x7e", "FPucA9l+"},
	testpair{"\x14\xfb\x9c\x03\xd9", "FPucA9k="},
	testpair{"\x14\xfb\x9c\x03", "FPucAw=="},

	// RFC 4648 examples
	testpair{"", ""},
	testpair{"f", "Zg=="},
	testpair{"fo", "Zm8="},
	testpair{"foo", "Zm9v"},
	testpair{"foob", "Zm9vYg=="},
	testpair{"fooba", "Zm9vYmE="},
	testpair{"foobar", "Zm9vYmFy"},

	// Wikipedia examples
	testpair{"sure.", "c3VyZS4="},
	testpair{"sure", "c3VyZQ=="},
	testpair{"sur", "c3Vy"},
	testpair{"su", "c3U="},
	testpair{"leasure.", "bGVhc3VyZS4="},
	testpair{"easure.", "ZWFzdXJlLg=="},
	testpair{"asure.", "YXN1cmUu"},
	testpair{"sure.", "c3VyZS4="},
}

var bigtest = testpair {
	"Twas brillig, and the slithy toves",
	"VHdhcyBicmlsbGlnLCBhbmQgdGhlIHNsaXRoeSB0b3Zlcw=="
}

func testEqual(t *testing.T, msg string, args ...) bool {
	v := reflect.NewValue(args).(*reflect.StructValue);
	v1 := v.Field(v.NumField() - 2);
	v2 := v.Field(v.NumField() - 1);
	if v1.Interface() != v2.Interface() {
		t.Errorf(msg, args);
		return false;
	}
	return true;
}

func TestEncode(t *testing.T) {
	for _, p := range pairs {
		buf := make([]byte, StdEncoding.EncodedLen(len(p.decoded)));
		StdEncoding.Encode(strings.Bytes(p.decoded), buf);
		testEqual(t, "Encode(%q) = %q, want %q", p.decoded, string(buf), p.encoded);
	}
}

func TestEncoder(t *testing.T) {
	for _, p := range pairs {
		bb := &bytes.Buffer{};
		encoder := NewEncoder(StdEncoding, bb);
		encoder.Write(strings.Bytes(p.decoded));
		encoder.Close();
		testEqual(t, "Encode(%q) = %q, want %q", p.decoded, bb.String(), p.encoded);
	}
}

func TestEncoderBuffering(t *testing.T) {
	input := strings.Bytes(bigtest.decoded);
	for bs := 1; bs <= 12; bs++ {
		bb := &bytes.Buffer{};
		encoder := NewEncoder(StdEncoding, bb);
		for pos := 0; pos < len(input); pos += bs {
			end := pos+bs;
			if end > len(input) {
				end = len(input);
			}
			n, err := encoder.Write(input[pos:end]);
			testEqual(t, "Write(%q) gave error %v, want %v", input[pos:end], err, os.Error(nil));
			testEqual(t, "Write(%q) gave length %v, want %v", input[pos:end], n, end-pos);
		}
		err := encoder.Close();
		testEqual(t, "Close gave error %v, want %v", err, os.Error(nil));
		testEqual(t, "Encoding/%d of %q = %q, want %q", bs, bigtest.decoded, bb.String(), bigtest.encoded);
	}
}

func TestDecode(t *testing.T) {
	for _, p := range pairs {
		dbuf := make([]byte, StdEncoding.DecodedLen(len(p.encoded)));
		count, end, err := StdEncoding.decode(strings.Bytes(p.encoded), dbuf);
		testEqual(t, "Decode(%q) = error %v, want %v", p.encoded, err, os.Error(nil));
		testEqual(t, "Decode(%q) = length %v, want %v", p.encoded, count, len(p.decoded));
		if len(p.encoded) > 0 {
			testEqual(t, "Decode(%q) = end %v, want %v", p.encoded, end, (p.encoded[len(p.encoded)-1] == '='));
		}
		testEqual(t, "Decode(%q) = %q, want %q", p.encoded, string(dbuf[0:count]), p.decoded);
	}
}

func TestDecoder(t *testing.T) {
	for _, p := range pairs {
		decoder := NewDecoder(StdEncoding, strings.NewBuffer(p.encoded));
		dbuf := make([]byte, StdEncoding.DecodedLen(len(p.encoded)));
		count, err := decoder.Read(dbuf);
		if err != nil && err != os.EOF {
			t.Fatal("Read failed", err);
		}
		testEqual(t, "Read from %q = length %v, want %v", p.encoded, count, len(p.decoded));
		testEqual(t, "Decoding of %q = %q, want %q", p.encoded, string(dbuf[0:count]), p.decoded);
		if err != os.EOF {
			count, err = decoder.Read(dbuf);
		}
		testEqual(t, "Read from %q = %v, want %v", p.encoded, err, os.EOF);
	}
}

func TestDecoderBuffering(t *testing.T) {
	for bs := 1; bs <= 12; bs++ {
		decoder := NewDecoder(StdEncoding, strings.NewBuffer(bigtest.encoded));
		buf := make([]byte, len(bigtest.decoded) + 12);
		var total int;
		for total = 0; total < len(bigtest.decoded); {
			n, err := decoder.Read(buf[total:total+bs]);
			testEqual(t, "Read from %q at pos %d = %d, %v, want _, %v", bigtest.encoded, total, n, err, os.Error(nil));
			total += n;
		}
		testEqual(t, "Decoding/%d of %q = %q, want %q", bs, bigtest.encoded, string(buf[0:total]), bigtest.decoded);
	}
}

func TestDecodeCorrupt(t *testing.T) {
	type corrupt struct {
		e string;
		p int;
	};
	examples := []corrupt {
		corrupt{"!!!!", 0},
		corrupt{"x===", 1},
		corrupt{"AA=A", 2},
		corrupt{"AAA=AAAA", 3},
		corrupt{"AAAAA", 4},
		corrupt{"AAAAAA", 4},
	};

	for _, e := range examples {
		dbuf := make([]byte, StdEncoding.DecodedLen(len(e.e)));
		_, err := StdEncoding.Decode(strings.Bytes(e.e), dbuf);
		switch err := err.(type) {
		case CorruptInputError:
			testEqual(t, "Corruption in %q at offset %v, want %v", e.e, int(err), e.p);
		default:
			t.Error("Decoder failed to detect corruption in", e);
		}
	}
}

func TestBig(t *testing.T) {
	n := 3*1000+1;
	raw := make([]byte, n);
	const alpha = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
	for i := 0; i < n; i++ {
		raw[i] = alpha[i%len(alpha)];
	}
	encoded := new(bytes.Buffer);
	w := NewEncoder(StdEncoding, encoded);
	nn, err := w.Write(raw);
	if nn != n || err != nil {
		t.Fatalf("Encoder.Write(raw) = %d, %v want %d, nil", nn, err, n);
	}
	err = w.Close();
	if err != nil {
		t.Fatalf("Encoder.Close() = %v want nil", err);
	}
	decoded, err := io.ReadAll(NewDecoder(StdEncoding, encoded));
	if err != nil {
		t.Fatalf("io.ReadAll(NewDecoder(...)): %v", err);
	}

	if !bytes.Equal(raw, decoded) {
		var i int;
		for i = 0; i < len(decoded) && i < len(raw); i++ {
			if decoded[i] != raw[i] {
				break;
			}
		}
		t.Errorf("Decode(Encode(%d-byte string)) failed at offset %d", n, i);
	}
}
