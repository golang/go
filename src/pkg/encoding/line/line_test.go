// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package line

import (
	"bytes"
	"os"
	"testing"
)

var testOutput = []byte("0123456789abcdefghijklmnopqrstuvwxy")
var testInput = []byte("012\n345\n678\n9ab\ncde\nfgh\nijk\nlmn\nopq\nrst\nuvw\nxy")
var testInputrn = []byte("012\r\n345\r\n678\r\n9ab\r\ncde\r\nfgh\r\nijk\r\nlmn\r\nopq\r\nrst\r\nuvw\r\nxy\r\n\n\r\n")

// TestReader wraps a []byte and returns reads of a specific length.
type testReader struct {
	data   []byte
	stride int
}

func (t *testReader) Read(buf []byte) (n int, err os.Error) {
	n = t.stride
	if n > len(t.data) {
		n = len(t.data)
	}
	if n > len(buf) {
		n = len(buf)
	}
	copy(buf, t.data)
	t.data = t.data[n:]
	if len(t.data) == 0 {
		err = os.EOF
	}
	return
}

func testLineReader(t *testing.T, input []byte) {
	for stride := 1; stride < len(input); stride++ {
		done := 0
		reader := testReader{input, stride}
		l := NewReader(&reader, len(input)+1)
		for {
			line, isPrefix, err := l.ReadLine()
			if len(line) > 0 && err != nil {
				t.Errorf("ReadLine returned both data and error: %s", err)
			}
			if isPrefix {
				t.Errorf("ReadLine returned prefix")
			}
			if err != nil {
				if err != os.EOF {
					t.Fatalf("Got unknown error: %s", err)
				}
				break
			}
			if want := testOutput[done : done+len(line)]; !bytes.Equal(want, line) {
				t.Errorf("Bad line at stride %d: want: %x got: %x", stride, want, line)
			}
			done += len(line)
		}
		if done != len(testOutput) {
			t.Error("ReadLine didn't return everything")
		}
	}
}

func TestReader(t *testing.T) {
	testLineReader(t, testInput)
	testLineReader(t, testInputrn)
}

func TestLineTooLong(t *testing.T) {
	buf := bytes.NewBuffer([]byte("aaabbbcc\n"))
	l := NewReader(buf, 3)
	line, isPrefix, err := l.ReadLine()
	if !isPrefix || !bytes.Equal(line, []byte("aaa")) || err != nil {
		t.Errorf("bad result for first line: %x %s", line, err)
	}
	line, isPrefix, err = l.ReadLine()
	if !isPrefix || !bytes.Equal(line, []byte("bbb")) || err != nil {
		t.Errorf("bad result for second line: %x", line)
	}
	line, isPrefix, err = l.ReadLine()
	if isPrefix || !bytes.Equal(line, []byte("cc")) || err != nil {
		t.Errorf("bad result for third line: %x", line)
	}
}
