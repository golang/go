// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io

import (
	"io";
	"rand";
	"testing";
)


const N = 10000;  // make this bigger for a larger (and slower) test
var data []byte;  // test data for write tests


func init() {
	data = make([]byte, N);
	for i := 0; i < len(data); i++ {
		data[i] = 'a' + byte(i % 26)
	}
}


// Verify that contents of buf match the string s.
func check(t *testing.T, testname string, buf *ByteBuffer, s string) {
	if buf.Len() != len(buf.Data()) {
		t.Errorf("%s: buf.Len() == %d, len(buf.Data()) == %d\n", testname, buf.Len(), len(buf.Data()))
	}

	if buf.Len() != len(s) {
		t.Errorf("%s: buf.Len() == %d, len(s) == %d\n", testname, buf.Len(), len(s))
	}

	if string(buf.Data()) != s {
		t.Errorf("%s: string(buf.Data()) == %q, s == %q\n", testname, string(buf.Data()), s)
	}
}


// Fill buf through n writes of fub.
// The initial contents of buf corresponds to the string s;
// the result is the final contents of buf returned as a string.
func fill(t *testing.T, testname string, buf *ByteBuffer, s string, n int, fub []byte) string {
	check(t, testname + " (fill 1)", buf, s);
	for ; n > 0; n-- {
		m, err := buf.Write(fub);
		if m != len(fub) {
			t.Errorf(testname + " (fill 2): m == %d, expected %d\n", m, len(fub));
		}
		if err != nil {
			t.Errorf(testname + " (fill 3): err should always be nil, found err == %s\n", err);
		}
		s += string(fub);
		check(t, testname + " (fill 4)", buf, s);
	}
	return s;
}


// Empty buf through repeated reads into fub.
// The initial contents of buf corresponds to the string s.
func empty(t *testing.T, testname string, buf *ByteBuffer, s string, fub []byte) {
	check(t, testname + " (empty 1)", buf, s);

	for {
		n, err := buf.Read(fub);
		if n == 0 {
			break;
		}
		if err != nil {
			t.Errorf(testname + " (empty 2): err should always be nil, found err == %s\n", err);
		}
		s = s[n : len(s)];
		check(t, testname + " (empty 3)", buf, s);
	}

	check(t, testname + " (empty 4)", buf, "");
}


func TestBasicOperations(t *testing.T) {
	var buf ByteBuffer;

	for i := 0; i < 5; i++ {
		check(t, "TestBasicOperations (1)", &buf, "");

		buf.Reset();
		check(t, "TestBasicOperations (2)", &buf, "");

		buf.Truncate(0);
		check(t, "TestBasicOperations (3)", &buf, "");

		n, err := buf.Write(data[0 : 1]);
		if n != 1 {
			t.Errorf("wrote 1 byte, but n == %d\n", n);
		}
		if err != nil {
			t.Errorf("err should always be nil, but err == %s\n", err);
		}
		check(t, "TestBasicOperations (4)", &buf, "a");

		buf.WriteByte(data[1]);
		check(t, "TestBasicOperations (5)", &buf, "ab");

		n, err = buf.Write(data[2 : 26]);
		if n != 24 {
			t.Errorf("wrote 25 bytes, but n == %d\n", n);
		}
		check(t, "TestBasicOperations (6)", &buf, string(data[0 : 26]));

		buf.Truncate(26);
		check(t, "TestBasicOperations (7)", &buf, string(data[0 : 26]));

		buf.Truncate(20);
		check(t, "TestBasicOperations (8)", &buf, string(data[0 : 20]));

		empty(t, "TestBasicOperations (9)", &buf, string(data[0 : 20]), make([]byte, 5));
		empty(t, "TestBasicOperations (10)", &buf, "", make([]byte, 100));
	}
}


func TestLargeWrites(t *testing.T) {
	var buf ByteBuffer;
	for i := 3; i < 30; i += 3 {
		s := fill(t, "TestLargeWrites (1)", &buf, "", 5, data);
		empty(t, "TestLargeWrites (2)", &buf, s, make([]byte, len(data)/i));
	}
	check(t, "TestLargeWrites (3)", &buf, "");
}


func TestLargeReads(t *testing.T) {
	var buf ByteBuffer;
	for i := 3; i < 30; i += 3 {
		s := fill(t, "TestLargeReads (1)", &buf, "", 5, data[0 : len(data)/i]);
		empty(t, "TestLargeReads (2)", &buf, s, make([]byte, len(data)));
	}
	check(t, "TestLargeReads (3)", &buf, "");
}


func TestMixedReadsAndWrites(t *testing.T) {
	var buf ByteBuffer;
	s := "";
	for i := 0; i < 50; i++ {
		wlen := rand.Intn(len(data));
		s = fill(t, "TestMixedReadsAndWrites (1)", &buf, s, 1, data[0 : wlen]);

		rlen := rand.Intn(len(data));
		fub := make([]byte, rlen);
		n, err := buf.Read(fub);
		s = s[n : len(s)];
	}
	empty(t, "TestMixedReadsAndWrites (2)", &buf, s, make([]byte, buf.Len()));
}
