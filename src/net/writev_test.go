// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js
// +build !js

package net

import (
	"bytes"
	"fmt"
	"internal/poll"
	"io"
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
	got, err := io.ReadAll(&buffers)
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
	oldHook := poll.TestHookDidWritev
	defer func() { poll.TestHookDidWritev = oldHook }()
	var writeLog struct {
		sync.Mutex
		log []int
	}
	poll.TestHookDidWritev = func(size int) {
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
		all, err := io.ReadAll(c)
		if !bytes.Equal(all, want.Bytes()) || err != nil {
			return fmt.Errorf("client read %q, %v; want %q, nil", all, err, want.Bytes())
		}

		writeLog.Lock() // no need to unlock
		var gotSum int
		for _, v := range writeLog.log {
			gotSum += v
		}

		var wantSum int
		switch runtime.GOOS {
		case "android", "darwin", "ios", "dragonfly", "freebsd", "illumos", "linux", "netbsd", "openbsd":
			var wantMinCalls int
			wantSum = want.Len()
			v := chunks
			for v > 0 {
				wantMinCalls++
				v -= 1024
			}
			if len(writeLog.log) < wantMinCalls {
				t.Errorf("write calls = %v < wanted min %v", len(writeLog.log), wantMinCalls)
			}
		case "windows":
			var wantCalls int
			wantSum = want.Len()
			if wantSum > 0 {
				wantCalls = 1 // windows will always do 1 syscall, unless sending empty buffer
			}
			if len(writeLog.log) != wantCalls {
				t.Errorf("write calls = %v; want %v", len(writeLog.log), wantCalls)
			}
		}
		if gotSum != wantSum {
			t.Errorf("writev call sum  = %v; want %v", gotSum, wantSum)
		}
		return nil
	})
}

func TestWritevError(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skipf("skipping the test: windows does not have problem sending large chunks of data")
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	ch := make(chan Conn, 1)
	go func() {
		defer close(ch)
		c, err := ln.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		ch <- c
	}()
	c1, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c1.Close()
	c2 := <-ch
	if c2 == nil {
		t.Fatal("no server side connection")
	}
	c2.Close()

	// 1 GB of data should be enough to notice the connection is gone.
	// Just a few bytes is not enough.
	// Arrange to reuse the same 1 MB buffer so that we don't allocate much.
	buf := make([]byte, 1<<20)
	buffers := make(Buffers, 1<<10)
	for i := range buffers {
		buffers[i] = buf
	}
	if _, err := buffers.WriteTo(c1); err == nil {
		t.Fatal("Buffers.WriteTo(closed conn) succeeded, want error")
	}
}
