// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hpack

import (
	"bufio"
	"bytes"
	"encoding/hex"
	"fmt"
	"math/rand"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestStaticTable(t *testing.T) {
	fromSpec := `
          +-------+-----------------------------+---------------+
          | 1     | :authority                  |               |
          | 2     | :method                     | GET           |
          | 3     | :method                     | POST          |
          | 4     | :path                       | /             |
          | 5     | :path                       | /index.html   |
          | 6     | :scheme                     | http          |
          | 7     | :scheme                     | https         |
          | 8     | :status                     | 200           |
          | 9     | :status                     | 204           |
          | 10    | :status                     | 206           |
          | 11    | :status                     | 304           |
          | 12    | :status                     | 400           |
          | 13    | :status                     | 404           |
          | 14    | :status                     | 500           |
          | 15    | accept-charset              |               |
          | 16    | accept-encoding             | gzip, deflate |
          | 17    | accept-language             |               |
          | 18    | accept-ranges               |               |
          | 19    | accept                      |               |
          | 20    | access-control-allow-origin |               |
          | 21    | age                         |               |
          | 22    | allow                       |               |
          | 23    | authorization               |               |
          | 24    | cache-control               |               |
          | 25    | content-disposition         |               |
          | 26    | content-encoding            |               |
          | 27    | content-language            |               |
          | 28    | content-length              |               |
          | 29    | content-location            |               |
          | 30    | content-range               |               |
          | 31    | content-type                |               |
          | 32    | cookie                      |               |
          | 33    | date                        |               |
          | 34    | etag                        |               |
          | 35    | expect                      |               |
          | 36    | expires                     |               |
          | 37    | from                        |               |
          | 38    | host                        |               |
          | 39    | if-match                    |               |
          | 40    | if-modified-since           |               |
          | 41    | if-none-match               |               |
          | 42    | if-range                    |               |
          | 43    | if-unmodified-since         |               |
          | 44    | last-modified               |               |
          | 45    | link                        |               |
          | 46    | location                    |               |
          | 47    | max-forwards                |               |
          | 48    | proxy-authenticate          |               |
          | 49    | proxy-authorization         |               |
          | 50    | range                       |               |
          | 51    | referer                     |               |
          | 52    | refresh                     |               |
          | 53    | retry-after                 |               |
          | 54    | server                      |               |
          | 55    | set-cookie                  |               |
          | 56    | strict-transport-security   |               |
          | 57    | transfer-encoding           |               |
          | 58    | user-agent                  |               |
          | 59    | vary                        |               |
          | 60    | via                         |               |
          | 61    | www-authenticate            |               |
          +-------+-----------------------------+---------------+
`
	bs := bufio.NewScanner(strings.NewReader(fromSpec))
	re := regexp.MustCompile(`\| (\d+)\s+\| (\S+)\s*\| (\S(.*\S)?)?\s+\|`)
	for bs.Scan() {
		l := bs.Text()
		if !strings.Contains(l, "|") {
			continue
		}
		m := re.FindStringSubmatch(l)
		if m == nil {
			continue
		}
		i, err := strconv.Atoi(m[1])
		if err != nil {
			t.Errorf("Bogus integer on line %q", l)
			continue
		}
		if i < 1 || i > len(staticTable) {
			t.Errorf("Bogus index %d on line %q", i, l)
			continue
		}
		if got, want := staticTable[i-1].Name, m[2]; got != want {
			t.Errorf("header index %d name = %q; want %q", i, got, want)
		}
		if got, want := staticTable[i-1].Value, m[3]; got != want {
			t.Errorf("header index %d value = %q; want %q", i, got, want)
		}
	}
	if err := bs.Err(); err != nil {
		t.Error(err)
	}
}

func (d *Decoder) mustAt(idx int) HeaderField {
	if hf, ok := d.at(uint64(idx)); !ok {
		panic(fmt.Sprintf("bogus index %d", idx))
	} else {
		return hf
	}
}

func TestDynamicTableAt(t *testing.T) {
	d := NewDecoder(4096, nil)
	at := d.mustAt
	if got, want := at(2), (pair(":method", "GET")); got != want {
		t.Errorf("at(2) = %v; want %v", got, want)
	}
	d.dynTab.add(pair("foo", "bar"))
	d.dynTab.add(pair("blake", "miz"))
	if got, want := at(len(staticTable)+1), (pair("blake", "miz")); got != want {
		t.Errorf("at(dyn 1) = %v; want %v", got, want)
	}
	if got, want := at(len(staticTable)+2), (pair("foo", "bar")); got != want {
		t.Errorf("at(dyn 2) = %v; want %v", got, want)
	}
	if got, want := at(3), (pair(":method", "POST")); got != want {
		t.Errorf("at(3) = %v; want %v", got, want)
	}
}

func TestDynamicTableSearch(t *testing.T) {
	dt := dynamicTable{}
	dt.setMaxSize(4096)

	dt.add(pair("foo", "bar"))
	dt.add(pair("blake", "miz"))
	dt.add(pair(":method", "GET"))

	tests := []struct {
		hf        HeaderField
		wantI     uint64
		wantMatch bool
	}{
		// Name and Value match
		{pair("foo", "bar"), 3, true},
		{pair(":method", "GET"), 1, true},

		// Only name match because of Sensitive == true
		{HeaderField{"blake", "miz", true}, 2, false},

		// Only Name matches
		{pair("foo", "..."), 3, false},
		{pair("blake", "..."), 2, false},
		{pair(":method", "..."), 1, false},

		// None match
		{pair("foo-", "bar"), 0, false},
	}
	for _, tt := range tests {
		if gotI, gotMatch := dt.search(tt.hf); gotI != tt.wantI || gotMatch != tt.wantMatch {
			t.Errorf("d.search(%+v) = %v, %v; want %v, %v", tt.hf, gotI, gotMatch, tt.wantI, tt.wantMatch)
		}
	}
}

func TestDynamicTableSizeEvict(t *testing.T) {
	d := NewDecoder(4096, nil)
	if want := uint32(0); d.dynTab.size != want {
		t.Fatalf("size = %d; want %d", d.dynTab.size, want)
	}
	add := d.dynTab.add
	add(pair("blake", "eats pizza"))
	if want := uint32(15 + 32); d.dynTab.size != want {
		t.Fatalf("after pizza, size = %d; want %d", d.dynTab.size, want)
	}
	add(pair("foo", "bar"))
	if want := uint32(15 + 32 + 6 + 32); d.dynTab.size != want {
		t.Fatalf("after foo bar, size = %d; want %d", d.dynTab.size, want)
	}
	d.dynTab.setMaxSize(15 + 32 + 1 /* slop */)
	if want := uint32(6 + 32); d.dynTab.size != want {
		t.Fatalf("after setMaxSize, size = %d; want %d", d.dynTab.size, want)
	}
	if got, want := d.mustAt(len(staticTable)+1), (pair("foo", "bar")); got != want {
		t.Errorf("at(dyn 1) = %v; want %v", got, want)
	}
	add(pair("long", strings.Repeat("x", 500)))
	if want := uint32(0); d.dynTab.size != want {
		t.Fatalf("after big one, size = %d; want %d", d.dynTab.size, want)
	}
}

func TestDecoderDecode(t *testing.T) {
	tests := []struct {
		name       string
		in         []byte
		want       []HeaderField
		wantDynTab []HeaderField // newest entry first
	}{
		// C.2.1 Literal Header Field with Indexing
		// http://http2.github.io/http2-spec/compression.html#rfc.section.C.2.1
		{"C.2.1", dehex("400a 6375 7374 6f6d 2d6b 6579 0d63 7573 746f 6d2d 6865 6164 6572"),
			[]HeaderField{pair("custom-key", "custom-header")},
			[]HeaderField{pair("custom-key", "custom-header")},
		},

		// C.2.2 Literal Header Field without Indexing
		// http://http2.github.io/http2-spec/compression.html#rfc.section.C.2.2
		{"C.2.2", dehex("040c 2f73 616d 706c 652f 7061 7468"),
			[]HeaderField{pair(":path", "/sample/path")},
			[]HeaderField{}},

		// C.2.3 Literal Header Field never Indexed
		// http://http2.github.io/http2-spec/compression.html#rfc.section.C.2.3
		{"C.2.3", dehex("1008 7061 7373 776f 7264 0673 6563 7265 74"),
			[]HeaderField{{"password", "secret", true}},
			[]HeaderField{}},

		// C.2.4 Indexed Header Field
		// http://http2.github.io/http2-spec/compression.html#rfc.section.C.2.4
		{"C.2.4", []byte("\x82"),
			[]HeaderField{pair(":method", "GET")},
			[]HeaderField{}},
	}
	for _, tt := range tests {
		d := NewDecoder(4096, nil)
		hf, err := d.DecodeFull(tt.in)
		if err != nil {
			t.Errorf("%s: %v", tt.name, err)
			continue
		}
		if !reflect.DeepEqual(hf, tt.want) {
			t.Errorf("%s: Got %v; want %v", tt.name, hf, tt.want)
		}
		gotDynTab := d.dynTab.reverseCopy()
		if !reflect.DeepEqual(gotDynTab, tt.wantDynTab) {
			t.Errorf("%s: dynamic table after = %v; want %v", tt.name, gotDynTab, tt.wantDynTab)
		}
	}
}

func (dt *dynamicTable) reverseCopy() (hf []HeaderField) {
	hf = make([]HeaderField, len(dt.ents))
	for i := range hf {
		hf[i] = dt.ents[len(dt.ents)-1-i]
	}
	return
}

type encAndWant struct {
	enc         []byte
	want        []HeaderField
	wantDynTab  []HeaderField
	wantDynSize uint32
}

// C.3 Request Examples without Huffman Coding
// http://http2.github.io/http2-spec/compression.html#rfc.section.C.3
func TestDecodeC3_NoHuffman(t *testing.T) {
	testDecodeSeries(t, 4096, []encAndWant{
		{dehex("8286 8441 0f77 7777 2e65 7861 6d70 6c65 2e63 6f6d"),
			[]HeaderField{
				pair(":method", "GET"),
				pair(":scheme", "http"),
				pair(":path", "/"),
				pair(":authority", "www.example.com"),
			},
			[]HeaderField{
				pair(":authority", "www.example.com"),
			},
			57,
		},
		{dehex("8286 84be 5808 6e6f 2d63 6163 6865"),
			[]HeaderField{
				pair(":method", "GET"),
				pair(":scheme", "http"),
				pair(":path", "/"),
				pair(":authority", "www.example.com"),
				pair("cache-control", "no-cache"),
			},
			[]HeaderField{
				pair("cache-control", "no-cache"),
				pair(":authority", "www.example.com"),
			},
			110,
		},
		{dehex("8287 85bf 400a 6375 7374 6f6d 2d6b 6579 0c63 7573 746f 6d2d 7661 6c75 65"),
			[]HeaderField{
				pair(":method", "GET"),
				pair(":scheme", "https"),
				pair(":path", "/index.html"),
				pair(":authority", "www.example.com"),
				pair("custom-key", "custom-value"),
			},
			[]HeaderField{
				pair("custom-key", "custom-value"),
				pair("cache-control", "no-cache"),
				pair(":authority", "www.example.com"),
			},
			164,
		},
	})
}

// C.4 Request Examples with Huffman Coding
// http://http2.github.io/http2-spec/compression.html#rfc.section.C.4
func TestDecodeC4_Huffman(t *testing.T) {
	testDecodeSeries(t, 4096, []encAndWant{
		{dehex("8286 8441 8cf1 e3c2 e5f2 3a6b a0ab 90f4 ff"),
			[]HeaderField{
				pair(":method", "GET"),
				pair(":scheme", "http"),
				pair(":path", "/"),
				pair(":authority", "www.example.com"),
			},
			[]HeaderField{
				pair(":authority", "www.example.com"),
			},
			57,
		},
		{dehex("8286 84be 5886 a8eb 1064 9cbf"),
			[]HeaderField{
				pair(":method", "GET"),
				pair(":scheme", "http"),
				pair(":path", "/"),
				pair(":authority", "www.example.com"),
				pair("cache-control", "no-cache"),
			},
			[]HeaderField{
				pair("cache-control", "no-cache"),
				pair(":authority", "www.example.com"),
			},
			110,
		},
		{dehex("8287 85bf 4088 25a8 49e9 5ba9 7d7f 8925 a849 e95b b8e8 b4bf"),
			[]HeaderField{
				pair(":method", "GET"),
				pair(":scheme", "https"),
				pair(":path", "/index.html"),
				pair(":authority", "www.example.com"),
				pair("custom-key", "custom-value"),
			},
			[]HeaderField{
				pair("custom-key", "custom-value"),
				pair("cache-control", "no-cache"),
				pair(":authority", "www.example.com"),
			},
			164,
		},
	})
}

// http://http2.github.io/http2-spec/compression.html#rfc.section.C.5
// "This section shows several consecutive header lists, corresponding
// to HTTP responses, on the same connection. The HTTP/2 setting
// parameter SETTINGS_HEADER_TABLE_SIZE is set to the value of 256
// octets, causing some evictions to occur."
func TestDecodeC5_ResponsesNoHuff(t *testing.T) {
	testDecodeSeries(t, 256, []encAndWant{
		{dehex(`
4803 3330 3258 0770 7269 7661 7465 611d
4d6f 6e2c 2032 3120 4f63 7420 3230 3133
2032 303a 3133 3a32 3120 474d 546e 1768
7474 7073 3a2f 2f77 7777 2e65 7861 6d70
6c65 2e63 6f6d
`),
			[]HeaderField{
				pair(":status", "302"),
				pair("cache-control", "private"),
				pair("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
				pair("location", "https://www.example.com"),
			},
			[]HeaderField{
				pair("location", "https://www.example.com"),
				pair("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
				pair("cache-control", "private"),
				pair(":status", "302"),
			},
			222,
		},
		{dehex("4803 3330 37c1 c0bf"),
			[]HeaderField{
				pair(":status", "307"),
				pair("cache-control", "private"),
				pair("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
				pair("location", "https://www.example.com"),
			},
			[]HeaderField{
				pair(":status", "307"),
				pair("location", "https://www.example.com"),
				pair("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
				pair("cache-control", "private"),
			},
			222,
		},
		{dehex(`
88c1 611d 4d6f 6e2c 2032 3120 4f63 7420
3230 3133 2032 303a 3133 3a32 3220 474d
54c0 5a04 677a 6970 7738 666f 6f3d 4153
444a 4b48 514b 425a 584f 5157 454f 5049
5541 5851 5745 4f49 553b 206d 6178 2d61
6765 3d33 3630 303b 2076 6572 7369 6f6e
3d31
`),
			[]HeaderField{
				pair(":status", "200"),
				pair("cache-control", "private"),
				pair("date", "Mon, 21 Oct 2013 20:13:22 GMT"),
				pair("location", "https://www.example.com"),
				pair("content-encoding", "gzip"),
				pair("set-cookie", "foo=ASDJKHQKBZXOQWEOPIUAXQWEOIU; max-age=3600; version=1"),
			},
			[]HeaderField{
				pair("set-cookie", "foo=ASDJKHQKBZXOQWEOPIUAXQWEOIU; max-age=3600; version=1"),
				pair("content-encoding", "gzip"),
				pair("date", "Mon, 21 Oct 2013 20:13:22 GMT"),
			},
			215,
		},
	})
}

// http://http2.github.io/http2-spec/compression.html#rfc.section.C.6
// "This section shows the same examples as the previous section, but
// using Huffman encoding for the literal values. The HTTP/2 setting
// parameter SETTINGS_HEADER_TABLE_SIZE is set to the value of 256
// octets, causing some evictions to occur. The eviction mechanism
// uses the length of the decoded literal values, so the same
// evictions occurs as in the previous section."
func TestDecodeC6_ResponsesHuffman(t *testing.T) {
	testDecodeSeries(t, 256, []encAndWant{
		{dehex(`
4882 6402 5885 aec3 771a 4b61 96d0 7abe
9410 54d4 44a8 2005 9504 0b81 66e0 82a6
2d1b ff6e 919d 29ad 1718 63c7 8f0b 97c8
e9ae 82ae 43d3
`),
			[]HeaderField{
				pair(":status", "302"),
				pair("cache-control", "private"),
				pair("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
				pair("location", "https://www.example.com"),
			},
			[]HeaderField{
				pair("location", "https://www.example.com"),
				pair("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
				pair("cache-control", "private"),
				pair(":status", "302"),
			},
			222,
		},
		{dehex("4883 640e ffc1 c0bf"),
			[]HeaderField{
				pair(":status", "307"),
				pair("cache-control", "private"),
				pair("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
				pair("location", "https://www.example.com"),
			},
			[]HeaderField{
				pair(":status", "307"),
				pair("location", "https://www.example.com"),
				pair("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
				pair("cache-control", "private"),
			},
			222,
		},
		{dehex(`
88c1 6196 d07a be94 1054 d444 a820 0595
040b 8166 e084 a62d 1bff c05a 839b d9ab
77ad 94e7 821d d7f2 e6c7 b335 dfdf cd5b
3960 d5af 2708 7f36 72c1 ab27 0fb5 291f
9587 3160 65c0 03ed 4ee5 b106 3d50 07
`),
			[]HeaderField{
				pair(":status", "200"),
				pair("cache-control", "private"),
				pair("date", "Mon, 21 Oct 2013 20:13:22 GMT"),
				pair("location", "https://www.example.com"),
				pair("content-encoding", "gzip"),
				pair("set-cookie", "foo=ASDJKHQKBZXOQWEOPIUAXQWEOIU; max-age=3600; version=1"),
			},
			[]HeaderField{
				pair("set-cookie", "foo=ASDJKHQKBZXOQWEOPIUAXQWEOIU; max-age=3600; version=1"),
				pair("content-encoding", "gzip"),
				pair("date", "Mon, 21 Oct 2013 20:13:22 GMT"),
			},
			215,
		},
	})
}

func testDecodeSeries(t *testing.T, size uint32, steps []encAndWant) {
	d := NewDecoder(size, nil)
	for i, step := range steps {
		hf, err := d.DecodeFull(step.enc)
		if err != nil {
			t.Fatalf("Error at step index %d: %v", i, err)
		}
		if !reflect.DeepEqual(hf, step.want) {
			t.Fatalf("At step index %d: Got headers %v; want %v", i, hf, step.want)
		}
		gotDynTab := d.dynTab.reverseCopy()
		if !reflect.DeepEqual(gotDynTab, step.wantDynTab) {
			t.Errorf("After step index %d, dynamic table = %v; want %v", i, gotDynTab, step.wantDynTab)
		}
		if d.dynTab.size != step.wantDynSize {
			t.Errorf("After step index %d, dynamic table size = %v; want %v", i, d.dynTab.size, step.wantDynSize)
		}
	}
}

func TestHuffmanDecode(t *testing.T) {
	tests := []struct {
		inHex, want string
	}{
		{"f1e3 c2e5 f23a 6ba0 ab90 f4ff", "www.example.com"},
		{"a8eb 1064 9cbf", "no-cache"},
		{"25a8 49e9 5ba9 7d7f", "custom-key"},
		{"25a8 49e9 5bb8 e8b4 bf", "custom-value"},
		{"6402", "302"},
		{"aec3 771a 4b", "private"},
		{"d07a be94 1054 d444 a820 0595 040b 8166 e082 a62d 1bff", "Mon, 21 Oct 2013 20:13:21 GMT"},
		{"9d29 ad17 1863 c78f 0b97 c8e9 ae82 ae43 d3", "https://www.example.com"},
		{"9bd9 ab", "gzip"},
		{"94e7 821d d7f2 e6c7 b335 dfdf cd5b 3960 d5af 2708 7f36 72c1 ab27 0fb5 291f 9587 3160 65c0 03ed 4ee5 b106 3d50 07",
			"foo=ASDJKHQKBZXOQWEOPIUAXQWEOIU; max-age=3600; version=1"},
	}
	for i, tt := range tests {
		var buf bytes.Buffer
		in, err := hex.DecodeString(strings.Replace(tt.inHex, " ", "", -1))
		if err != nil {
			t.Errorf("%d. hex input error: %v", i, err)
			continue
		}
		if _, err := HuffmanDecode(&buf, in); err != nil {
			t.Errorf("%d. decode error: %v", i, err)
			continue
		}
		if got := buf.String(); tt.want != got {
			t.Errorf("%d. decode = %q; want %q", i, got, tt.want)
		}
	}
}

func TestAppendHuffmanString(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"www.example.com", "f1e3 c2e5 f23a 6ba0 ab90 f4ff"},
		{"no-cache", "a8eb 1064 9cbf"},
		{"custom-key", "25a8 49e9 5ba9 7d7f"},
		{"custom-value", "25a8 49e9 5bb8 e8b4 bf"},
		{"302", "6402"},
		{"private", "aec3 771a 4b"},
		{"Mon, 21 Oct 2013 20:13:21 GMT", "d07a be94 1054 d444 a820 0595 040b 8166 e082 a62d 1bff"},
		{"https://www.example.com", "9d29 ad17 1863 c78f 0b97 c8e9 ae82 ae43 d3"},
		{"gzip", "9bd9 ab"},
		{"foo=ASDJKHQKBZXOQWEOPIUAXQWEOIU; max-age=3600; version=1",
			"94e7 821d d7f2 e6c7 b335 dfdf cd5b 3960 d5af 2708 7f36 72c1 ab27 0fb5 291f 9587 3160 65c0 03ed 4ee5 b106 3d50 07"},
	}
	for i, tt := range tests {
		buf := []byte{}
		want := strings.Replace(tt.want, " ", "", -1)
		buf = AppendHuffmanString(buf, tt.in)
		if got := hex.EncodeToString(buf); want != got {
			t.Errorf("%d. encode = %q; want %q", i, got, want)
		}
	}
}

func TestHuffmanMaxStrLen(t *testing.T) {
	const msg = "Some string"
	huff := AppendHuffmanString(nil, msg)

	testGood := func(max int) {
		var out bytes.Buffer
		if err := huffmanDecode(&out, max, huff); err != nil {
			t.Errorf("For maxLen=%d, unexpected error: %v", max, err)
		}
		if out.String() != msg {
			t.Errorf("For maxLen=%d, out = %q; want %q", max, out.String(), msg)
		}
	}
	testGood(0)
	testGood(len(msg))
	testGood(len(msg) + 1)

	var out bytes.Buffer
	if err := huffmanDecode(&out, len(msg)-1, huff); err != ErrStringLength {
		t.Errorf("err = %v; want ErrStringLength", err)
	}
}

func TestHuffmanRoundtripStress(t *testing.T) {
	const Len = 50 // of uncompressed string
	input := make([]byte, Len)
	var output bytes.Buffer
	var huff []byte

	n := 5000
	if testing.Short() {
		n = 100
	}
	seed := time.Now().UnixNano()
	t.Logf("Seed = %v", seed)
	src := rand.New(rand.NewSource(seed))
	var encSize int64
	for i := 0; i < n; i++ {
		for l := range input {
			input[l] = byte(src.Intn(256))
		}
		huff = AppendHuffmanString(huff[:0], string(input))
		encSize += int64(len(huff))
		output.Reset()
		if err := huffmanDecode(&output, 0, huff); err != nil {
			t.Errorf("Failed to decode %q -> %q -> error %v", input, huff, err)
			continue
		}
		if !bytes.Equal(output.Bytes(), input) {
			t.Errorf("Roundtrip failure on %q -> %q -> %q", input, huff, output.Bytes())
		}
	}
	t.Logf("Compressed size of original: %0.02f%% (%v -> %v)", 100*(float64(encSize)/(Len*float64(n))), Len*n, encSize)
}

func TestHuffmanDecodeFuzz(t *testing.T) {
	const Len = 50 // of compressed
	var buf, zbuf bytes.Buffer

	n := 5000
	if testing.Short() {
		n = 100
	}
	seed := time.Now().UnixNano()
	t.Logf("Seed = %v", seed)
	src := rand.New(rand.NewSource(seed))
	numFail := 0
	for i := 0; i < n; i++ {
		zbuf.Reset()
		if i == 0 {
			// Start with at least one invalid one.
			zbuf.WriteString("00\x91\xff\xff\xff\xff\xc8")
		} else {
			for l := 0; l < Len; l++ {
				zbuf.WriteByte(byte(src.Intn(256)))
			}
		}

		buf.Reset()
		if err := huffmanDecode(&buf, 0, zbuf.Bytes()); err != nil {
			if err == ErrInvalidHuffman {
				numFail++
				continue
			}
			t.Errorf("Failed to decode %q: %v", zbuf.Bytes(), err)
			continue
		}
	}
	t.Logf("%0.02f%% are invalid (%d / %d)", 100*float64(numFail)/float64(n), numFail, n)
	if numFail < 1 {
		t.Error("expected at least one invalid huffman encoding (test starts with one)")
	}
}

func TestReadVarInt(t *testing.T) {
	type res struct {
		i        uint64
		consumed int
		err      error
	}
	tests := []struct {
		n    byte
		p    []byte
		want res
	}{
		// Fits in a byte:
		{1, []byte{0}, res{0, 1, nil}},
		{2, []byte{2}, res{2, 1, nil}},
		{3, []byte{6}, res{6, 1, nil}},
		{4, []byte{14}, res{14, 1, nil}},
		{5, []byte{30}, res{30, 1, nil}},
		{6, []byte{62}, res{62, 1, nil}},
		{7, []byte{126}, res{126, 1, nil}},
		{8, []byte{254}, res{254, 1, nil}},

		// Doesn't fit in a byte:
		{1, []byte{1}, res{0, 0, errNeedMore}},
		{2, []byte{3}, res{0, 0, errNeedMore}},
		{3, []byte{7}, res{0, 0, errNeedMore}},
		{4, []byte{15}, res{0, 0, errNeedMore}},
		{5, []byte{31}, res{0, 0, errNeedMore}},
		{6, []byte{63}, res{0, 0, errNeedMore}},
		{7, []byte{127}, res{0, 0, errNeedMore}},
		{8, []byte{255}, res{0, 0, errNeedMore}},

		// Ignoring top bits:
		{5, []byte{255, 154, 10}, res{1337, 3, nil}}, // high dummy three bits: 111
		{5, []byte{159, 154, 10}, res{1337, 3, nil}}, // high dummy three bits: 100
		{5, []byte{191, 154, 10}, res{1337, 3, nil}}, // high dummy three bits: 101

		// Extra byte:
		{5, []byte{191, 154, 10, 2}, res{1337, 3, nil}}, // extra byte

		// Short a byte:
		{5, []byte{191, 154}, res{0, 0, errNeedMore}},

		// integer overflow:
		{1, []byte{255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}, res{0, 0, errVarintOverflow}},
	}
	for _, tt := range tests {
		i, remain, err := readVarInt(tt.n, tt.p)
		consumed := len(tt.p) - len(remain)
		got := res{i, consumed, err}
		if got != tt.want {
			t.Errorf("readVarInt(%d, %v ~ %x) = %+v; want %+v", tt.n, tt.p, tt.p, got, tt.want)
		}
	}
}

// Fuzz crash, originally reported at https://github.com/bradfitz/http2/issues/56
func TestHuffmanFuzzCrash(t *testing.T) {
	got, err := HuffmanDecodeToString([]byte("00\x91\xff\xff\xff\xff\xc8"))
	if got != "" {
		t.Errorf("Got %q; want empty string", got)
	}
	if err != ErrInvalidHuffman {
		t.Errorf("Err = %v; want ErrInvalidHuffman", err)
	}
}

func dehex(s string) []byte {
	s = strings.Replace(s, " ", "", -1)
	s = strings.Replace(s, "\n", "", -1)
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

func TestEmitEnabled(t *testing.T) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	enc.WriteField(HeaderField{Name: "foo", Value: "bar"})
	enc.WriteField(HeaderField{Name: "foo", Value: "bar"})

	numCallback := 0
	var dec *Decoder
	dec = NewDecoder(8<<20, func(HeaderField) {
		numCallback++
		dec.SetEmitEnabled(false)
	})
	if !dec.EmitEnabled() {
		t.Errorf("initial emit enabled = false; want true")
	}
	if _, err := dec.Write(buf.Bytes()); err != nil {
		t.Error(err)
	}
	if numCallback != 1 {
		t.Errorf("num callbacks = %d; want 1", numCallback)
	}
	if dec.EmitEnabled() {
		t.Errorf("emit enabled = true; want false")
	}
}

func TestSaveBufLimit(t *testing.T) {
	const maxStr = 1 << 10
	var got []HeaderField
	dec := NewDecoder(initialHeaderTableSize, func(hf HeaderField) {
		got = append(got, hf)
	})
	dec.SetMaxStringLength(maxStr)
	var frag []byte
	frag = append(frag[:0], encodeTypeByte(false, false))
	frag = appendVarInt(frag, 7, 3)
	frag = append(frag, "foo"...)
	frag = appendVarInt(frag, 7, 3)
	frag = append(frag, "bar"...)

	if _, err := dec.Write(frag); err != nil {
		t.Fatal(err)
	}

	want := []HeaderField{{Name: "foo", Value: "bar"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("After small writes, got %v; want %v", got, want)
	}

	frag = append(frag[:0], encodeTypeByte(false, false))
	frag = appendVarInt(frag, 7, maxStr*3)
	frag = append(frag, make([]byte, maxStr*3)...)

	_, err := dec.Write(frag)
	if err != ErrStringLength {
		t.Fatalf("Write error = %v; want ErrStringLength", err)
	}
}
