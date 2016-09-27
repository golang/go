// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"reflect"
	"runtime"
	"sync"
	"testing"
)

func TestBuffers_read(t *testing.T) {
	const story = "once upon a time in Gopherland ... "
	buffers := Buffers{
		[]byte("once "),
		[]byte("upon "),
		[]byte("a "),
		[]byte("time "),
		[]byte("in "),
		[]byte("Gopherland ... "),
	}
	got, err := ioutil.ReadAll(&buffers)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != story {
		t.Errorf("read %q; want %q", got, story)
	}
	if len(buffers) != 0 {
		t.Errorf("len(buffers) = %d; want 0", len(buffers))
	}
}

func TestBuffers_consume(t *testing.T) {
	tests := []struct {
		in      Buffers
		consume int64
		want    Buffers
	}{
		{
			in:      Buffers{[]byte("foo"), []byte("bar")},
			consume: 0,
			want:    Buffers{[]byte("foo"), []byte("bar")},
		},
		{
			in:      Buffers{[]byte("foo"), []byte("bar")},
			consume: 2,
			want:    Buffers{[]byte("o"), []byte("bar")},
		},
		{
			in:      Buffers{[]byte("foo"), []byte("bar")},
			consume: 3,
			want:    Buffers{[]byte("bar")},
		},
		{
			in:      Buffers{[]byte("foo"), []byte("bar")},
			consume: 4,
			want:    Buffers{[]byte("ar")},
		},
		{
			in:      Buffers{nil, nil, nil, []byte("bar")},
			consume: 1,
			want:    Buffers{[]byte("ar")},
		},
		{
			in:      Buffers{nil, nil, nil, []byte("foo")},
			consume: 0,
			want:    Buffers{[]byte("foo")},
		},
		{
			in:      Buffers{nil, nil, nil},
			consume: 0,
			want:    Buffers{},
		},
	}
	for i, tt := range tests {
		in := tt.in
		in.consume(tt.consume)
		if !reflect.DeepEqual(in, tt.want) {
			t.Errorf("%d. after consume(%d) = %+v, want %+v", i, tt.consume, in, tt.want)
		}
	}
}

func TestBuffers_WriteTo(t *testing.T) {
	for _, name := range []string{"WriteTo", "Copy"} {
		for _, size := range []int{0, 10, 1023, 1024, 1025} {
			t.Run(fmt.Sprintf("%s/%d", name, size), func(t *testing.T) {
				testBuffer_writeTo(t, size, name == "Copy")
			})
		}
	}
}

func testBuffer_writeTo(t *testing.T, chunks int, useCopy bool) {
	oldHook := testHookDidWritev
	defer func() { testHookDidWritev = oldHook }()
	var writeLog struct {
		sync.Mutex
		log []int
	}
	testHookDidWritev = func(size int) {
		writeLog.Lock()
		writeLog.log = append(writeLog.log, size)
		writeLog.Unlock()
	}
	var want bytes.Buffer
	for i := 0; i < chunks; i++ {
		want.WriteByte(byte(i))
	}

	withTCPConnPair(t, func(c *TCPConn) error {
		buffers := make(Buffers, chunks)
		for i := range buffers {
			buffers[i] = want.Bytes()[i : i+1]
		}
		var n int64
		var err error
		if useCopy {
			n, err = io.Copy(c, &buffers)
		} else {
			n, err = buffers.WriteTo(c)
		}
		if err != nil {
			return err
		}
		if len(buffers) != 0 {
			return fmt.Errorf("len(buffers) = %d; want 0", len(buffers))
		}
		if n != int64(want.Len()) {
			return fmt.Errorf("Buffers.WriteTo returned %d; want %d", n, want.Len())
		}
		return nil
	}, func(c *TCPConn) error {
		all, err := ioutil.ReadAll(c)
		if !bytes.Equal(all, want.Bytes()) || err != nil {
			return fmt.Errorf("client read %q, %v; want %q, nil", all, err, want.Bytes())
		}

		writeLog.Lock() // no need to unlock
		var gotSum int
		for _, v := range writeLog.log {
			gotSum += v
		}

		var wantSum int
		var wantMinCalls int
		switch runtime.GOOS {
		case "darwin", "dragonfly", "freebsd", "linux", "netbsd", "openbsd":
			wantSum = want.Len()
			v := chunks
			for v > 0 {
				wantMinCalls++
				v -= 1024
			}
		}
		if len(writeLog.log) < wantMinCalls {
			t.Errorf("write calls = %v < wanted min %v", len(writeLog.log), wantMinCalls)
		}
		if gotSum != wantSum {
			t.Errorf("writev call sum  = %v; want %v", gotSum, wantSum)
		}
		return nil
	})
}
