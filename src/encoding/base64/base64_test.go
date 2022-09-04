// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base64

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"
	"runtime/debug"
	"strings"
	"testing"
	"time"
)

type testpair struct {
	decoded, encoded string
}

var pairs = []testpair{
	// RFC 3548 examples
	{"\x14\xfb\x9c\x03\xd9\x7e", "FPucA9l+"},
	{"\x14\xfb\x9c\x03\xd9", "FPucA9k="},
	{"\x14\xfb\x9c\x03", "FPucAw=="},

	// RFC 4648 examples
	{"", ""},
	{"f", "Zg=="},
	{"fo", "Zm8="},
	{"foo", "Zm9v"},
	{"foob", "Zm9vYg=="},
	{"fooba", "Zm9vYmE="},
	{"foobar", "Zm9vYmFy"},

	// Wikipedia examples
	{"sure.", "c3VyZS4="},
	{"sure", "c3VyZQ=="},
	{"sur", "c3Vy"},
	{"su", "c3U="},
	{"leasure.", "bGVhc3VyZS4="},
	{"easure.", "ZWFzdXJlLg=="},
	{"asure.", "YXN1cmUu"},
	{"sure.", "c3VyZS4="},
}

// Do nothing to a reference base64 string (leave in standard format)
func stdRef(ref string) string {
	return ref
}

// Convert a reference string to URL-encoding
func urlRef(ref string) string {
	ref = strings.ReplaceAll(ref, "+", "-")
	ref = strings.ReplaceAll(ref, "/", "_")
	return ref
}

// Convert a reference string to raw, unpadded format
func rawRef(ref string) string {
	return strings.TrimRight(ref, "=")
}

// Both URL and unpadding conversions
func rawURLRef(ref string) string {
	return rawRef(urlRef(ref))
}

// A nonstandard encoding with a funny padding character, for testing
var funnyEncoding = NewEncoding(encodeStd).WithPadding(rune('@'))

func funnyRef(ref string) string {
	return strings.ReplaceAll(ref, "=", "@")
}

type encodingTest struct {
	enc  *Encoding           // Encoding to test
	conv func(string) string // Reference string converter
}

var encodingTests = []encodingTest{
	{StdEncoding, stdRef},
	{URLEncoding, urlRef},
	{RawStdEncoding, rawRef},
	{RawURLEncoding, rawURLRef},
	{funnyEncoding, funnyRef},
	{StdEncoding.Strict(), stdRef},
	{URLEncoding.Strict(), urlRef},
	{RawStdEncoding.Strict(), rawRef},
	{RawURLEncoding.Strict(), rawURLRef},
	{funnyEncoding.Strict(), funnyRef},
}

var bigtest = testpair{
	"Twas brillig, and the slithy toves",
	"VHdhcyBicmlsbGlnLCBhbmQgdGhlIHNsaXRoeSB0b3Zlcw==",
}

func testEqual(t *testing.T, msg string, args ...any) bool {
	t.Helper()
	if args[len(args)-2] != args[len(args)-1] {
		t.Errorf(msg, args...)
		return false
	}
	return true
}

func TestEncode(t *testing.T) {
	for _, p := range pairs {
		for _, tt := range encodingTests {
			got := tt.enc.EncodeToString([]byte(p.decoded))
			testEqual(t, "Encode(%q) = %q, want %q", p.decoded,
				got, tt.conv(p.encoded))
		}
	}
}

func TestEncoder(t *testing.T) {
	for _, p := range pairs {
		bb := &strings.Builder{}
		encoder := NewEncoder(StdEncoding, bb)
		encoder.Write([]byte(p.decoded))
		encoder.Close()
		testEqual(t, "Encode(%q) = %q, want %q", p.decoded, bb.String(), p.encoded)
	}
}

func TestEncoderBuffering(t *testing.T) {
	input := []byte(bigtest.decoded)
	for bs := 1; bs <= 12; bs++ {
		bb := &strings.Builder{}
		encoder := NewEncoder(StdEncoding, bb)
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
		testEqual(t, "Encoding/%d of %q = %q, want %q", bs, bigtest.decoded, bb.String(), bigtest.encoded)
	}
}

func TestDecode(t *testing.T) {
	for _, p := range pairs {
		for _, tt := range encodingTests {
			encoded := tt.conv(p.encoded)
			dbuf := make([]byte, tt.enc.DecodedLen(len(encoded)))
			count, err := tt.enc.Decode(dbuf, []byte(encoded))
			testEqual(t, "Decode(%q) = error %v, want %v", encoded, err, error(nil))
			testEqual(t, "Decode(%q) = length %v, want %v", encoded, count, len(p.decoded))
			testEqual(t, "Decode(%q) = %q, want %q", encoded, string(dbuf[0:count]), p.decoded)

			dbuf, err = tt.enc.DecodeString(encoded)
			testEqual(t, "DecodeString(%q) = error %v, want %v", encoded, err, error(nil))
			testEqual(t, "DecodeString(%q) = %q, want %q", encoded, string(dbuf), p.decoded)
		}
	}
}

func TestDecoder(t *testing.T) {
	for _, p := range pairs {
		decoder := NewDecoder(StdEncoding, strings.NewReader(p.encoded))
		dbuf := make([]byte, StdEncoding.DecodedLen(len(p.encoded)))
		count, err := decoder.Read(dbuf)
		if err != nil && err != io.EOF {
			t.Fatal("Read failed", err)
		}
		testEqual(t, "Read from %q = length %v, want %v", p.encoded, count, len(p.decoded))
		testEqual(t, "Decoding of %q = %q, want %q", p.encoded, string(dbuf[0:count]), p.decoded)
		if err != io.EOF {
			_, err = decoder.Read(dbuf)
		}
		testEqual(t, "Read from %q = %v, want %v", p.encoded, err, io.EOF)
	}
}

func TestDecoderBuffering(t *testing.T) {
	for bs := 1; bs <= 12; bs++ {
		decoder := NewDecoder(StdEncoding, strings.NewReader(bigtest.encoded))
		buf := make([]byte, len(bigtest.decoded)+12)
		var total int
		var n int
		var err error
		for total = 0; total < len(bigtest.decoded) && err == nil; {
			n, err = decoder.Read(buf[total : total+bs])
			total += n
		}
		if err != nil && err != io.EOF {
			t.Errorf("Read from %q at pos %d = %d, unexpected error %v", bigtest.encoded, total, n, err)
		}
		testEqual(t, "Decoding/%d of %q = %q, want %q", bs, bigtest.encoded, string(buf[0:total]), bigtest.decoded)
	}
}

func TestDecodeCorrupt(t *testing.T) {
	testCases := []struct {
		input  string
		offset int // -1 means no corruption.
	}{
		{"", -1},
		{"\n", -1},
		{"AAA=\n", -1},
		{"AAAA\n", -1},
		{"!!!!", 0},
		{"====", 0},
		{"x===", 1},
		{"=AAA", 0},
		{"A=AA", 1},
		{"AA=A", 2},
		{"AA==A", 4},
		{"AAA=AAAA", 4},
		{"AAAAA", 4},
		{"AAAAAA", 4},
		{"A=", 1},
		{"A==", 1},
		{"AA=", 3},
		{"AA==", -1},
		{"AAA=", -1},
		{"AAAA", -1},
		{"AAAAAA=", 7},
		{"YWJjZA=====", 8},
		{"A!\n", 1},
		{"A=\n", 1},
	}
	for _, tc := range testCases {
		dbuf := make([]byte, StdEncoding.DecodedLen(len(tc.input)))
		_, err := StdEncoding.Decode(dbuf, []byte(tc.input))
		if tc.offset == -1 {
			if err != nil {
				t.Error("Decoder wrongly detected corruption in", tc.input)
			}
			continue
		}
		switch err := err.(type) {
		case CorruptInputError:
			testEqual(t, "Corruption in %q at offset %v, want %v", tc.input, int(err), tc.offset)
		default:
			t.Error("Decoder failed to detect corruption in", tc)
		}
	}
}

func TestDecodeBounds(t *testing.T) {
	var buf [32]byte
	s := StdEncoding.EncodeToString(buf[:])
	defer func() {
		if err := recover(); err != nil {
			t.Fatalf("Decode panicked unexpectedly: %v\n%s", err, debug.Stack())
		}
	}()
	n, err := StdEncoding.Decode(buf[:], []byte(s))
	if n != len(buf) || err != nil {
		t.Fatalf("StdEncoding.Decode = %d, %v, want %d, nil", n, err, len(buf))
	}
}

func TestEncodedLen(t *testing.T) {
	for _, tt := range []struct {
		enc  *Encoding
		n    int
		want int
	}{
		{RawStdEncoding, 0, 0},
		{RawStdEncoding, 1, 2},
		{RawStdEncoding, 2, 3},
		{RawStdEncoding, 3, 4},
		{RawStdEncoding, 7, 10},
		{StdEncoding, 0, 0},
		{StdEncoding, 1, 4},
		{StdEncoding, 2, 4},
		{StdEncoding, 3, 4},
		{StdEncoding, 4, 8},
		{StdEncoding, 7, 12},
	} {
		if got := tt.enc.EncodedLen(tt.n); got != tt.want {
			t.Errorf("EncodedLen(%d): got %d, want %d", tt.n, got, tt.want)
		}
	}
}

func TestDecodedLen(t *testing.T) {
	for _, tt := range []struct {
		enc  *Encoding
		n    int
		want int
	}{
		{RawStdEncoding, 0, 0},
		{RawStdEncoding, 2, 1},
		{RawStdEncoding, 3, 2},
		{RawStdEncoding, 4, 3},
		{RawStdEncoding, 10, 7},
		{StdEncoding, 0, 0},
		{StdEncoding, 4, 3},
		{StdEncoding, 8, 6},
	} {
		if got := tt.enc.DecodedLen(tt.n); got != tt.want {
			t.Errorf("DecodedLen(%d): got %d, want %d", tt.n, got, tt.want)
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
	w := NewEncoder(StdEncoding, encoded)
	nn, err := w.Write(raw)
	if nn != n || err != nil {
		t.Fatalf("Encoder.Write(raw) = %d, %v want %d, nil", nn, err, n)
	}
	err = w.Close()
	if err != nil {
		t.Fatalf("Encoder.Close() = %v want nil", err)
	}
	decoded, err := io.ReadAll(NewDecoder(StdEncoding, encoded))
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

func TestNewLineCharacters(t *testing.T) {
	// Each of these should decode to the string "sure", without errors.
	const expected = "sure"
	examples := []string{
		"c3VyZQ==",
		"c3VyZQ==\r",
		"c3VyZQ==\n",
		"c3VyZQ==\r\n",
		"c3VyZ\r\nQ==",
		"c3V\ryZ\nQ==",
		"c3V\nyZ\rQ==",
		"c3VyZ\nQ==",
		"c3VyZQ\n==",
		"c3VyZQ=\n=",
		"c3VyZQ=\r\n\r\n=",
	}
	for _, e := range examples {
		buf, err := StdEncoding.DecodeString(e)
		if err != nil {
			t.Errorf("Decode(%q) failed: %v", e, err)
			continue
		}
		if s := string(buf); s != expected {
			t.Errorf("Decode(%q) = %q, want %q", e, s, expected)
		}
	}
}

type nextRead struct {
	n   int   // bytes to return
	err error // error to return
}

// faultInjectReader returns data from source, rate-limited
// and with the errors as written to nextc.
type faultInjectReader struct {
	source string
	nextc  <-chan nextRead
}

func (r *faultInjectReader) Read(p []byte) (int, error) {
	nr := <-r.nextc
	if len(p) > nr.n {
		p = p[:nr.n]
	}
	n := copy(p, r.source)
	r.source = r.source[n:]
	return n, nr.err
}

// tests that we don't ignore errors from our underlying reader
func TestDecoderIssue3577(t *testing.T) {
	next := make(chan nextRead, 10)
	wantErr := errors.New("my error")
	next <- nextRead{5, nil}
	next <- nextRead{10, wantErr}
	next <- nextRead{0, wantErr}
	d := NewDecoder(StdEncoding, &faultInjectReader{
		source: "VHdhcyBicmlsbGlnLCBhbmQgdGhlIHNsaXRoeSB0b3Zlcw==", // twas brillig...
		nextc:  next,
	})
	errc := make(chan error, 1)
	go func() {
		_, err := io.ReadAll(d)
		errc <- err
	}()
	select {
	case err := <-errc:
		if err != wantErr {
			t.Errorf("got error %v; want %v", err, wantErr)
		}
	case <-time.After(5 * time.Second):
		t.Errorf("timeout; Decoder blocked without returning an error")
	}
}

func TestDecoderIssue4779(t *testing.T) {
	encoded := `CP/EAT8AAAEF
AQEBAQEBAAAAAAAAAAMAAQIEBQYHCAkKCwEAAQUBAQEBAQEAAAAAAAAAAQACAwQFBgcICQoLEAAB
BAEDAgQCBQcGCAUDDDMBAAIRAwQhEjEFQVFhEyJxgTIGFJGhsUIjJBVSwWIzNHKC0UMHJZJT8OHx
Y3M1FqKygyZEk1RkRcKjdDYX0lXiZfKzhMPTdePzRieUpIW0lcTU5PSltcXV5fVWZnaGlqa2xtbm
9jdHV2d3h5ent8fX5/cRAAICAQIEBAMEBQYHBwYFNQEAAhEDITESBEFRYXEiEwUygZEUobFCI8FS
0fAzJGLhcoKSQ1MVY3M08SUGFqKygwcmNcLSRJNUoxdkRVU2dGXi8rOEw9N14/NGlKSFtJXE1OT0
pbXF1eX1VmZ2hpamtsbW5vYnN0dXZ3eHl6e3x//aAAwDAQACEQMRAD8A9VSSSSUpJJJJSkkkJ+Tj
1kiy1jCJJDnAcCTykpKkuQ6p/jN6FgmxlNduXawwAzaGH+V6jn/R/wCt71zdn+N/qL3kVYFNYB4N
ji6PDVjWpKp9TSXnvTf8bFNjg3qOEa2n6VlLpj/rT/pf567DpX1i6L1hs9Py67X8mqdtg/rUWbbf
+gkp0kkkklKSSSSUpJJJJT//0PVUkkklKVLq3WMDpGI7KzrNjADtYNXvI/Mqr/Pd/q9W3vaxjnvM
NaCXE9gNSvGPrf8AWS3qmba5jjsJhoB0DAf0NDf6sevf+/lf8Hj0JJATfWT6/dV6oXU1uOLQeKKn
EQP+Hubtfe/+R7Mf/g7f5xcocp++Z11JMCJPgFBxOg7/AOuqDx8I/ikpkXkmSdU8mJIJA/O8EMAy
j+mSARB/17pKVXYWHXjsj7yIex0PadzXMO1zT5KHoNA3HT8ietoGhgjsfA+CSnvvqh/jJtqsrwOv
2b6NGNzXfTYexzJ+nU7/ALkf4P8Awv6P9KvTQQ4AgyDqCF85Pho3CTB7eHwXoH+LT65uZbX9X+o2
bqbPb06551Y4
`
	encodedShort := strings.ReplaceAll(encoded, "\n", "")

	dec := NewDecoder(StdEncoding, strings.NewReader(encoded))
	res1, err := io.ReadAll(dec)
	if err != nil {
		t.Errorf("ReadAll failed: %v", err)
	}

	dec = NewDecoder(StdEncoding, strings.NewReader(encodedShort))
	var res2 []byte
	res2, err = io.ReadAll(dec)
	if err != nil {
		t.Errorf("ReadAll failed: %v", err)
	}

	if !bytes.Equal(res1, res2) {
		t.Error("Decoded results not equal")
	}
}

func TestDecoderIssue7733(t *testing.T) {
	s, err := StdEncoding.DecodeString("YWJjZA=====")
	want := CorruptInputError(8)
	if !reflect.DeepEqual(want, err) {
		t.Errorf("Error = %v; want CorruptInputError(8)", err)
	}
	if string(s) != "abcd" {
		t.Errorf("DecodeString = %q; want abcd", s)
	}
}

func TestDecoderIssue15656(t *testing.T) {
	_, err := StdEncoding.Strict().DecodeString("WvLTlMrX9NpYDQlEIFlnDB==")
	want := CorruptInputError(22)
	if !reflect.DeepEqual(want, err) {
		t.Errorf("Error = %v; want CorruptInputError(22)", err)
	}
	_, err = StdEncoding.Strict().DecodeString("WvLTlMrX9NpYDQlEIFlnDA==")
	if err != nil {
		t.Errorf("Error = %v; want nil", err)
	}
	_, err = StdEncoding.DecodeString("WvLTlMrX9NpYDQlEIFlnDB==")
	if err != nil {
		t.Errorf("Error = %v; want nil", err)
	}
}

func BenchmarkEncodeToString(b *testing.B) {
	data := make([]byte, 8192)
	b.SetBytes(int64(len(data)))
	for i := 0; i < b.N; i++ {
		StdEncoding.EncodeToString(data)
	}
}

func BenchmarkDecodeString(b *testing.B) {
	sizes := []int{2, 4, 8, 64, 8192}
	benchFunc := func(b *testing.B, benchSize int) {
		data := StdEncoding.EncodeToString(make([]byte, benchSize))
		b.SetBytes(int64(len(data)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			StdEncoding.DecodeString(data)
		}
	}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			benchFunc(b, size)
		})
	}
}

func BenchmarkNewEncoding(b *testing.B) {
	b.SetBytes(int64(len(Encoding{}.decodeMap)))
	for i := 0; i < b.N; i++ {
		e := NewEncoding(encodeStd)
		for _, v := range e.decodeMap {
			_ = v
		}
	}
}

func TestDecoderRaw(t *testing.T) {
	source := "AAAAAA"
	want := []byte{0, 0, 0, 0}

	// Direct.
	dec1, err := RawURLEncoding.DecodeString(source)
	if err != nil || !bytes.Equal(dec1, want) {
		t.Errorf("RawURLEncoding.DecodeString(%q) = %x, %v, want %x, nil", source, dec1, err, want)
	}

	// Through reader. Used to fail.
	r := NewDecoder(RawURLEncoding, bytes.NewReader([]byte(source)))
	dec2, err := io.ReadAll(io.LimitReader(r, 100))
	if err != nil || !bytes.Equal(dec2, want) {
		t.Errorf("reading NewDecoder(RawURLEncoding, %q) = %x, %v, want %x, nil", source, dec2, err, want)
	}

	// Should work with padding.
	r = NewDecoder(URLEncoding, bytes.NewReader([]byte(source+"==")))
	dec3, err := io.ReadAll(r)
	if err != nil || !bytes.Equal(dec3, want) {
		t.Errorf("reading NewDecoder(URLEncoding, %q) = %x, %v, want %x, nil", source+"==", dec3, err, want)
	}
}
