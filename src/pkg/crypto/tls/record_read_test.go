// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes";
	"testing";
	"testing/iotest";
)

func matchRecord(r1, r2 *record) bool {
	if (r1 == nil) != (r2 == nil) {
		return false
	}
	if r1 == nil {
		return true
	}
	return r1.contentType == r2.contentType &&
		r1.major == r2.major &&
		r1.minor == r2.minor &&
		bytes.Compare(r1.payload, r2.payload) == 0;
}

type recordReaderTest struct {
	in	[]byte;
	out	[]*record;
}

var recordReaderTests = []recordReaderTest{
	recordReaderTest{nil, nil},
	recordReaderTest{fromHex("01"), nil},
	recordReaderTest{fromHex("0102"), nil},
	recordReaderTest{fromHex("010203"), nil},
	recordReaderTest{fromHex("01020300"), nil},
	recordReaderTest{fromHex("0102030000"), []*record{&record{1, 2, 3, nil}}},
	recordReaderTest{fromHex("01020300000102030000"), []*record{&record{1, 2, 3, nil}, &record{1, 2, 3, nil}}},
	recordReaderTest{fromHex("0102030001fe0102030002feff"), []*record{&record{1, 2, 3, []byte{0xfe}}, &record{1, 2, 3, []byte{0xfe, 0xff}}}},
	recordReaderTest{fromHex("010203000001020300"), []*record{&record{1, 2, 3, nil}}},
}

func TestRecordReader(t *testing.T) {
	for i, test := range recordReaderTests {
		buf := bytes.NewBuffer(test.in);
		c := make(chan *record);
		go recordReader(c, buf);
		matchRecordReaderOutput(t, i, test, c);

		buf = bytes.NewBuffer(test.in);
		buf2 := iotest.OneByteReader(buf);
		c = make(chan *record);
		go recordReader(c, buf2);
		matchRecordReaderOutput(t, i*2, test, c);
	}
}

func matchRecordReaderOutput(t *testing.T, i int, test recordReaderTest, c <-chan *record) {
	for j, r1 := range test.out {
		r2 := <-c;
		if r2 == nil {
			t.Errorf("#%d truncated after %d values", i, j);
			break;
		}
		if !matchRecord(r1, r2) {
			t.Errorf("#%d (%d) got:%#v want:%#v", i, j, r2, r1)
		}
	}
	<-c;
	if !closed(c) {
		t.Errorf("#%d: channel didn't close", i)
	}
}
