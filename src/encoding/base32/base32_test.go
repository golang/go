// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base32

import (
	"bytes"
	"errors"
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

type testpair struct {
	decoded, encoded string
}

var pairs = []testpair{
	// RFC 4648 examples
	{"", ""},
	{"f", "MY======"},
	{"fo", "MZXQ===="},
	{"foo", "MZXW6==="},
	{"foob", "MZXW6YQ="},
	{"fooba", "MZXW6YTB"},
	{"foobar", "MZXW6YTBOI======"},

	// Wikipedia examples, converted to base32
	{"sure.", "ON2XEZJO"},
	{"sure", "ON2XEZI="},
	{"sur", "ON2XE==="},
	{"su", "ON2Q===="},
	{"leasure.", "NRSWC43VOJSS4==="},
	{"easure.", "MVQXG5LSMUXA===="},
	{"asure.", "MFZXK4TFFY======"},
	{"sure.", "ON2XEZJO"},
}

var bigtest = testpair{
	"Twas brillig, and the slithy toves",
	"KR3WC4ZAMJZGS3DMNFTSYIDBNZSCA5DIMUQHG3DJORUHSIDUN53GK4Y=",
}

func testEqual(t *testing.T, msg string, args ...interface{}) bool {
	t.Helper()
	if args[len(args)-2] != args[len(args)-1] {
		t.Errorf(msg, args...)
		return false
	}
	return true
}

func TestEncode(t *testing.T) {
	for _, p := range pairs {
		got := StdEncoding.EncodeToString([]byte(p.decoded))
		testEqual(t, "Encode(%q) = %q, want %q", p.decoded, got, p.encoded)
	}
}

func TestEncoder(t *testing.T) {
	for _, p := range pairs {
		bb := &bytes.Buffer{}
		encoder := NewEncoder(StdEncoding, bb)
		encoder.Write([]byte(p.decoded))
		encoder.Close()
		testEqual(t, "Encode(%q) = %q, want %q", p.decoded, bb.String(), p.encoded)
	}
}

func TestEncoderBuffering(t *testing.T) {
	input := []byte(bigtest.decoded)
	for bs := 1; bs <= 12; bs++ {
		bb := &bytes.Buffer{}
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
		dbuf := make([]byte, StdEncoding.DecodedLen(len(p.encoded)))
		count, end, err := StdEncoding.decode(dbuf, []byte(p.encoded))
		testEqual(t, "Decode(%q) = error %v, want %v", p.encoded, err, error(nil))
		testEqual(t, "Decode(%q) = length %v, want %v", p.encoded, count, len(p.decoded))
		if len(p.encoded) > 0 {
			testEqual(t, "Decode(%q) = end %v, want %v", p.encoded, end, (p.encoded[len(p.encoded)-1] == '='))
		}
		testEqual(t, "Decode(%q) = %q, want %q", p.encoded,
			string(dbuf[0:count]),
			p.decoded)

		dbuf, err = StdEncoding.DecodeString(p.encoded)
		testEqual(t, "DecodeString(%q) = error %v, want %v", p.encoded, err, error(nil))
		testEqual(t, "DecodeString(%q) = %q, want %q", p.encoded, string(dbuf), p.decoded)
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
			count, err = decoder.Read(dbuf)
		}
		testEqual(t, "Read from %q = %v, want %v", p.encoded, err, io.EOF)
	}
}

type badReader struct {
	data   []byte
	errs   []error
	called int
	limit  int
}

// Populates p with data, returns a count of the bytes written and an
// error.  The error returned is taken from badReader.errs, with each
// invocation of Read returning the next error in this slice, or io.EOF,
// if all errors from the slice have already been returned.  The
// number of bytes returned is determined by the size of the input buffer
// the test passes to decoder.Read and will be a multiple of 8, unless
// badReader.limit is non zero.
func (b *badReader) Read(p []byte) (int, error) {
	lim := len(p)
	if b.limit != 0 && b.limit < lim {
		lim = b.limit
	}
	if len(b.data) < lim {
		lim = len(b.data)
	}
	for i := range p[:lim] {
		p[i] = b.data[i]
	}
	b.data = b.data[lim:]
	err := io.EOF
	if b.called < len(b.errs) {
		err = b.errs[b.called]
	}
	b.called++
	return lim, err
}

// TestIssue20044 tests that decoder.Read behaves correctly when the caller
// supplied reader returns an error.
func TestIssue20044(t *testing.T) {
	badErr := errors.New("bad reader error")
	testCases := []struct {
		r       badReader
		res     string
		err     error
		dbuflen int
	}{
		// Check valid input data accompanied by an error is processed and the error is propagated.
		{r: badReader{data: []byte("MY======"), errs: []error{badErr}},
			res: "f", err: badErr},
		// Check a read error accompanied by input data consisting of newlines only is propagated.
		{r: badReader{data: []byte("\n\n\n\n\n\n\n\n"), errs: []error{badErr, nil}},
			res: "", err: badErr},
		// Reader will be called twice.  The first time it will return 8 newline characters.  The
		// second time valid base32 encoded data and an error.  The data should be decoded
		// correctly and the error should be propagated.
		{r: badReader{data: []byte("\n\n\n\n\n\n\n\nMY======"), errs: []error{nil, badErr}},
			res: "f", err: badErr, dbuflen: 8},
		// Reader returns invalid input data (too short) and an error.  Verify the reader
		// error is returned.
		{r: badReader{data: []byte("MY====="), errs: []error{badErr}},
			res: "", err: badErr},
		// Reader returns invalid input data (too short) but no error.  Verify io.ErrUnexpectedEOF
		// is returned.
		{r: badReader{data: []byte("MY====="), errs: []error{nil}},
			res: "", err: io.ErrUnexpectedEOF},
		// Reader returns invalid input data and an error.  Verify the reader and not the
		// decoder error is returned.
		{r: badReader{data: []byte("Ma======"), errs: []error{badErr}},
			res: "", err: badErr},
		// Reader returns valid data and io.EOF.  Check data is decoded and io.EOF is propagated.
		{r: badReader{data: []byte("MZXW6YTB"), errs: []error{io.EOF}},
			res: "fooba", err: io.EOF},
		// Check errors are properly reported when decoder.Read is called multiple times.
		// decoder.Read will be called 8 times, badReader.Read will be called twice, returning
		// valid data both times but an error on the second call.
		{r: badReader{data: []byte("NRSWC43VOJSS4==="), errs: []error{nil, badErr}},
			res: "leasure.", err: badErr, dbuflen: 1},
		// Check io.EOF is properly reported when decoder.Read is called multiple times.
		// decoder.Read will be called 8 times, badReader.Read will be called twice, returning
		// valid data both times but io.EOF on the second call.
		{r: badReader{data: []byte("NRSWC43VOJSS4==="), errs: []error{nil, io.EOF}},
			res: "leasure.", err: io.EOF, dbuflen: 1},
		// The following two test cases check that errors are propagated correctly when more than
		// 8 bytes are read at a time.
		{r: badReader{data: []byte("NRSWC43VOJSS4==="), errs: []error{io.EOF}},
			res: "leasure.", err: io.EOF, dbuflen: 11},
		{r: badReader{data: []byte("NRSWC43VOJSS4==="), errs: []error{badErr}},
			res: "leasure.", err: badErr, dbuflen: 11},
		// Check that errors are correctly propagated when the reader returns valid bytes in
		// groups that are not divisible by 8.  The first read will return 11 bytes and no
		// error.  The second will return 7 and an error.  The data should be decoded correctly
		// and the error should be propagated.
		{r: badReader{data: []byte("NRSWC43VOJSS4==="), errs: []error{nil, badErr}, limit: 11},
			res: "leasure.", err: badErr},
	}

	for _, tc := range testCases {
		input := tc.r.data
		decoder := NewDecoder(StdEncoding, &tc.r)
		var dbuflen int
		if tc.dbuflen > 0 {
			dbuflen = tc.dbuflen
		} else {
			dbuflen = StdEncoding.DecodedLen(len(input))
		}
		dbuf := make([]byte, dbuflen)
		var err error
		var res []byte
		for err == nil {
			var n int
			n, err = decoder.Read(dbuf)
			if n > 0 {
				res = append(res, dbuf[:n]...)
			}
		}

		testEqual(t, "Decoding of %q = %q, want %q", string(input), string(res), tc.res)
		testEqual(t, "Decoding of %q err = %v, expected %v", string(input), err, tc.err)
	}
}

// TestDecoderError verifies decode errors are propagated when there are no read
// errors.
func TestDecoderError(t *testing.T) {
	for _, readErr := range []error{io.EOF, nil} {
		input := "MZXW6YTb"
		dbuf := make([]byte, StdEncoding.DecodedLen(len(input)))
		br := badReader{data: []byte(input), errs: []error{readErr}}
		decoder := NewDecoder(StdEncoding, &br)
		n, err := decoder.Read(dbuf)
		testEqual(t, "Read after EOF, n = %d, expected %d", n, 0)
		if _, ok := err.(CorruptInputError); !ok {
			t.Errorf("Corrupt input error expected.  Found %T", err)
		}
	}
}

// TestReaderEOF ensures decoder.Read behaves correctly when input data is
// exhausted.
func TestReaderEOF(t *testing.T) {
	for _, readErr := range []error{io.EOF, nil} {
		input := "MZXW6YTB"
		br := badReader{data: []byte(input), errs: []error{nil, readErr}}
		decoder := NewDecoder(StdEncoding, &br)
		dbuf := make([]byte, StdEncoding.DecodedLen(len(input)))
		n, err := decoder.Read(dbuf)
		testEqual(t, "Decoding of %q err = %v, expected %v", string(input), err, error(nil))
		n, err = decoder.Read(dbuf)
		testEqual(t, "Read after EOF, n = %d, expected %d", n, 0)
		testEqual(t, "Read after EOF, err = %v, expected %v", err, io.EOF)
		n, err = decoder.Read(dbuf)
		testEqual(t, "Read after EOF, n = %d, expected %d", n, 0)
		testEqual(t, "Read after EOF, err = %v, expected %v", err, io.EOF)
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
		{"!!!!", 0},
		{"x===", 0},
		{"AA=A====", 2},
		{"AAA=AAAA", 3},
		{"MMMMMMMMM", 8},
		{"MMMMMM", 0},
		{"A=", 1},
		{"AA=", 3},
		{"AA==", 4},
		{"AA===", 5},
		{"AAAA=", 5},
		{"AAAA==", 6},
		{"AAAAA=", 6},
		{"AAAAA==", 7},
		{"A=======", 1},
		{"AA======", -1},
		{"AAA=====", 3},
		{"AAAA====", -1},
		{"AAAAA===", -1},
		{"AAAAAA==", 6},
		{"AAAAAAA=", -1},
		{"AAAAAAAA", -1},
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
	decoded, err := ioutil.ReadAll(NewDecoder(StdEncoding, encoded))
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

func testStringEncoding(t *testing.T, expected string, examples []string) {
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

func TestNewLineCharacters(t *testing.T) {
	// Each of these should decode to the string "sure", without errors.
	examples := []string{
		"ON2XEZI=",
		"ON2XEZI=\r",
		"ON2XEZI=\n",
		"ON2XEZI=\r\n",
		"ON2XEZ\r\nI=",
		"ON2X\rEZ\nI=",
		"ON2X\nEZ\rI=",
		"ON2XEZ\nI=",
		"ON2XEZI\n=",
	}
	testStringEncoding(t, "sure", examples)

	// Each of these should decode to the string "foobar", without errors.
	examples = []string{
		"MZXW6YTBOI======",
		"MZXW6YTBOI=\r\n=====",
	}
	testStringEncoding(t, "foobar", examples)
}

func TestDecoderIssue4779(t *testing.T) {
	encoded := `JRXXEZLNEBUXA43VNUQGI33MN5ZCA43JOQQGC3LFOQWCAY3PNZZWKY3UMV2HK4
RAMFSGS4DJONUWG2LOM4QGK3DJOQWCA43FMQQGI3YKMVUXK43NN5SCA5DFNVYG64RANFXGG2LENFSH
K3TUEB2XIIDMMFRG64TFEBSXIIDEN5WG64TFEBWWCZ3OMEQGC3DJOF2WCLRAKV2CAZLONFWQUYLEEB
WWS3TJNUQHMZLONFQW2LBAOF2WS4ZANZXXG5DSOVSCAZLYMVZGG2LUMF2GS33OEB2WY3DBNVRW6IDM
MFRG64TJOMQG42LTNEQHK5AKMFWGS4LVNFYCAZLYEBSWCIDDN5WW233EN4QGG33OONSXC5LBOQXCAR
DVNFZSAYLVORSSA2LSOVZGKIDEN5WG64RANFXAU4TFOBZGK2DFNZSGK4TJOQQGS3RAOZXWY5LQORQX
IZJAOZSWY2LUEBSXG43FEBRWS3DMOVWSAZDPNRXXEZJAMV2SAZTVM5UWC5BANZ2WY3DBBJYGC4TJMF
2HK4ROEBCXQY3FOB2GK5LSEBZWS3TUEBXWGY3BMVRWC5BAMN2XA2LEMF2GC5BANZXW4IDQOJXWSZDF
NZ2CYIDTOVXHIIDJNYFGG5LMOBQSA4LVNEQG6ZTGNFRWSYJAMRSXGZLSOVXHIIDNN5WGY2LUEBQW42
LNEBUWIIDFON2CA3DBMJXXE5LNFY==
====`
	encodedShort := strings.ReplaceAll(encoded, "\n", "")

	dec := NewDecoder(StdEncoding, strings.NewReader(encoded))
	res1, err := ioutil.ReadAll(dec)
	if err != nil {
		t.Errorf("ReadAll failed: %v", err)
	}

	dec = NewDecoder(StdEncoding, strings.NewReader(encodedShort))
	var res2 []byte
	res2, err = ioutil.ReadAll(dec)
	if err != nil {
		t.Errorf("ReadAll failed: %v", err)
	}

	if !bytes.Equal(res1, res2) {
		t.Error("Decoded results not equal")
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
	data := StdEncoding.EncodeToString(make([]byte, 8192))
	b.SetBytes(int64(len(data)))
	for i := 0; i < b.N; i++ {
		StdEncoding.DecodeString(data)
	}
}

func TestWithCustomPadding(t *testing.T) {
	for _, testcase := range pairs {
		defaultPadding := StdEncoding.EncodeToString([]byte(testcase.decoded))
		customPadding := StdEncoding.WithPadding('@').EncodeToString([]byte(testcase.decoded))
		expected := strings.ReplaceAll(defaultPadding, "=", "@")

		if expected != customPadding {
			t.Errorf("Expected custom %s, got %s", expected, customPadding)
		}
		if testcase.encoded != defaultPadding {
			t.Errorf("Expected %s, got %s", testcase.encoded, defaultPadding)
		}
	}
}

func TestWithoutPadding(t *testing.T) {
	for _, testcase := range pairs {
		defaultPadding := StdEncoding.EncodeToString([]byte(testcase.decoded))
		customPadding := StdEncoding.WithPadding(NoPadding).EncodeToString([]byte(testcase.decoded))
		expected := strings.TrimRight(defaultPadding, "=")

		if expected != customPadding {
			t.Errorf("Expected custom %s, got %s", expected, customPadding)
		}
		if testcase.encoded != defaultPadding {
			t.Errorf("Expected %s, got %s", testcase.encoded, defaultPadding)
		}
	}
}

func TestDecodeWithPadding(t *testing.T) {
	encodings := []*Encoding{
		StdEncoding,
		StdEncoding.WithPadding('-'),
		StdEncoding.WithPadding(NoPadding),
	}

	for i, enc := range encodings {
		for _, pair := range pairs {

			input := pair.decoded
			encoded := enc.EncodeToString([]byte(input))

			decoded, err := enc.DecodeString(encoded)
			if err != nil {
				t.Errorf("DecodeString Error for encoding %d (%q): %v", i, input, err)
			}

			if input != string(decoded) {
				t.Errorf("Unexpected result for encoding %d: got %q; want %q", i, decoded, input)
			}
		}
	}
}

func TestDecodeWithWrongPadding(t *testing.T) {
	encoded := StdEncoding.EncodeToString([]byte("foobar"))

	_, err := StdEncoding.WithPadding('-').DecodeString(encoded)
	if err == nil {
		t.Error("expected error")
	}

	_, err = StdEncoding.WithPadding(NoPadding).DecodeString(encoded)
	if err == nil {
		t.Error("expected error")
	}
}

func TestBufferedDecodingSameError(t *testing.T) {
	testcases := []struct {
		prefix            string
		chunkCombinations [][]string
		expected          error
	}{
		// NBSWY3DPO5XXE3DE == helloworld
		// Test with "ZZ" as extra input
		{"helloworld", [][]string{
			[]string{"NBSW", "Y3DP", "O5XX", "E3DE", "ZZ"},
			[]string{"NBSWY3DPO5XXE3DE", "ZZ"},
			[]string{"NBSWY3DPO5XXE3DEZZ"},
			[]string{"NBS", "WY3", "DPO", "5XX", "E3D", "EZZ"},
			[]string{"NBSWY3DPO5XXE3", "DEZZ"},
		}, io.ErrUnexpectedEOF},

		// Test with "ZZY" as extra input
		{"helloworld", [][]string{
			[]string{"NBSW", "Y3DP", "O5XX", "E3DE", "ZZY"},
			[]string{"NBSWY3DPO5XXE3DE", "ZZY"},
			[]string{"NBSWY3DPO5XXE3DEZZY"},
			[]string{"NBS", "WY3", "DPO", "5XX", "E3D", "EZZY"},
			[]string{"NBSWY3DPO5XXE3", "DEZZY"},
		}, io.ErrUnexpectedEOF},

		// Normal case, this is valid input
		{"helloworld", [][]string{
			[]string{"NBSW", "Y3DP", "O5XX", "E3DE"},
			[]string{"NBSWY3DPO5XXE3DE"},
			[]string{"NBS", "WY3", "DPO", "5XX", "E3D", "E"},
			[]string{"NBSWY3DPO5XXE3", "DE"},
		}, nil},

		// MZXW6YTB = fooba
		{"fooba", [][]string{
			[]string{"MZXW6YTBZZ"},
			[]string{"MZXW6YTBZ", "Z"},
			[]string{"MZXW6YTB", "ZZ"},
			[]string{"MZXW6YT", "BZZ"},
			[]string{"MZXW6Y", "TBZZ"},
			[]string{"MZXW6Y", "TB", "ZZ"},
			[]string{"MZXW6", "YTBZZ"},
			[]string{"MZXW6", "YTB", "ZZ"},
			[]string{"MZXW6", "YT", "BZZ"},
		}, io.ErrUnexpectedEOF},

		// Normal case, this is valid input
		{"fooba", [][]string{
			[]string{"MZXW6YTB"},
			[]string{"MZXW6YT", "B"},
			[]string{"MZXW6Y", "TB"},
			[]string{"MZXW6", "YTB"},
			[]string{"MZXW6", "YT", "B"},
			[]string{"MZXW", "6YTB"},
			[]string{"MZXW", "6Y", "TB"},
		}, nil},
	}

	for _, testcase := range testcases {
		for _, chunks := range testcase.chunkCombinations {
			pr, pw := io.Pipe()

			// Write the encoded chunks into the pipe
			go func() {
				for _, chunk := range chunks {
					pw.Write([]byte(chunk))
				}
				pw.Close()
			}()

			decoder := NewDecoder(StdEncoding, pr)
			_, err := ioutil.ReadAll(decoder)

			if err != testcase.expected {
				t.Errorf("Expected %v, got %v; case %s %+v", testcase.expected, err, testcase.prefix, chunks)
			}
		}
	}
}

func TestEncodedDecodedLen(t *testing.T) {
	type test struct {
		in      int
		wantEnc int
		wantDec int
	}
	data := bytes.Repeat([]byte("x"), 100)
	for _, test := range []struct {
		name  string
		enc   *Encoding
		cases []test
	}{
		{"StdEncoding", StdEncoding, []test{
			{0, 0, 0},
			{1, 8, 5},
			{5, 8, 5},
			{6, 16, 10},
			{10, 16, 10},
		}},
		{"NoPadding", StdEncoding.WithPadding(NoPadding), []test{
			{0, 0, 0},
			{1, 2, 1},
			{2, 4, 2},
			{5, 8, 5},
			{6, 10, 6},
			{7, 12, 7},
			{10, 16, 10},
			{11, 18, 11},
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			for _, tc := range test.cases {
				encLen := test.enc.EncodedLen(tc.in)
				decLen := test.enc.DecodedLen(encLen)
				enc := test.enc.EncodeToString(data[:tc.in])
				if len(enc) != encLen {
					t.Fatalf("EncodedLen(%d) = %d but encoded to %q (%d)", tc.in, encLen, enc, len(enc))
				}
				if encLen != tc.wantEnc {
					t.Fatalf("EncodedLen(%d) = %d; want %d", tc.in, encLen, tc.wantEnc)
				}
				if decLen != tc.wantDec {
					t.Fatalf("DecodedLen(%d) = %d; want %d", encLen, decLen, tc.wantDec)
				}
			}
		})
	}
}

func TestWithoutPaddingClose(t *testing.T) {
	encodings := []*Encoding{
		StdEncoding,
		StdEncoding.WithPadding(NoPadding),
	}

	for _, encoding := range encodings {
		for _, testpair := range pairs {

			var buf bytes.Buffer
			encoder := NewEncoder(encoding, &buf)
			encoder.Write([]byte(testpair.decoded))
			encoder.Close()

			expected := testpair.encoded
			if encoding.padChar == NoPadding {
				expected = strings.ReplaceAll(expected, "=", "")
			}

			res := buf.String()

			if res != expected {
				t.Errorf("Expected %s got %s; padChar=%d", expected, res, encoding.padChar)
			}
		}
	}
}

func TestDecodeReadAll(t *testing.T) {
	encodings := []*Encoding{
		StdEncoding,
		StdEncoding.WithPadding(NoPadding),
	}

	for _, pair := range pairs {
		for encIndex, encoding := range encodings {
			encoded := pair.encoded
			if encoding.padChar == NoPadding {
				encoded = strings.ReplaceAll(encoded, "=", "")
			}

			decReader, err := ioutil.ReadAll(NewDecoder(encoding, strings.NewReader(encoded)))
			if err != nil {
				t.Errorf("NewDecoder error: %v", err)
			}

			if pair.decoded != string(decReader) {
				t.Errorf("Expected %s got %s; Encoding %d", pair.decoded, decReader, encIndex)
			}
		}
	}
}

func TestDecodeSmallBuffer(t *testing.T) {
	encodings := []*Encoding{
		StdEncoding,
		StdEncoding.WithPadding(NoPadding),
	}

	for bufferSize := 1; bufferSize < 200; bufferSize++ {
		for _, pair := range pairs {
			for encIndex, encoding := range encodings {
				encoded := pair.encoded
				if encoding.padChar == NoPadding {
					encoded = strings.ReplaceAll(encoded, "=", "")
				}

				decoder := NewDecoder(encoding, strings.NewReader(encoded))

				var allRead []byte

				for {
					buf := make([]byte, bufferSize)
					n, err := decoder.Read(buf)
					allRead = append(allRead, buf[0:n]...)
					if err == io.EOF {
						break
					}
					if err != nil {
						t.Error(err)
					}
				}

				if pair.decoded != string(allRead) {
					t.Errorf("Expected %s got %s; Encoding %d; bufferSize %d", pair.decoded, allRead, encIndex, bufferSize)
				}
			}
		}
	}
}
