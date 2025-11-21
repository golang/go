// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"slices"
	"testing"
	"time"
	"unsafe"
)

func TestProfBuf(t *testing.T) {
	const hdrSize = 2

	write := func(t *testing.T, b *ProfBuf, tag unsafe.Pointer, now int64, hdr []uint64, stk []uintptr) {
		b.Write(&tag, now, hdr, stk)
	}
	read := func(t *testing.T, b *ProfBuf, data []uint64, tags []unsafe.Pointer) {
		rdata, rtags, eof := b.Read(ProfBufNonBlocking)
		if !slices.Equal(rdata, data) || !slices.Equal(rtags, tags) {
			t.Fatalf("unexpected profile read:\nhave data %#x\nwant data %#x\nhave tags %#x\nwant tags %#x", rdata, data, rtags, tags)
		}
		if eof {
			t.Fatalf("unexpected eof")
		}
	}
	readBlock := func(t *testing.T, b *ProfBuf, data []uint64, tags []unsafe.Pointer) func() {
		c := make(chan int)
		go func() {
			eof := data == nil
			rdata, rtags, reof := b.Read(ProfBufBlocking)
			if !slices.Equal(rdata, data) || !slices.Equal(rtags, tags) || reof != eof {
				// Errorf, not Fatalf, because called in goroutine.
				t.Errorf("unexpected profile read:\nhave data %#x\nwant data %#x\nhave tags %#x\nwant tags %#x\nhave eof=%v, want %v", rdata, data, rtags, tags, reof, eof)
			}
			c <- 1
		}()
		time.Sleep(10 * time.Millisecond) // let goroutine run and block
		return func() { <-c }
	}
	readEOF := func(t *testing.T, b *ProfBuf) {
		rdata, rtags, eof := b.Read(ProfBufBlocking)
		if rdata != nil || rtags != nil || !eof {
			t.Errorf("unexpected profile read: %#x, %#x, eof=%v; want nil, nil, eof=true", rdata, rtags, eof)
		}
		rdata, rtags, eof = b.Read(ProfBufNonBlocking)
		if rdata != nil || rtags != nil || !eof {
			t.Errorf("unexpected profile read (non-blocking): %#x, %#x, eof=%v; want nil, nil, eof=true", rdata, rtags, eof)
		}
	}

	myTags := make([]byte, 100)
	t.Logf("myTags is %p", &myTags[0])

	t.Run("BasicWriteRead", func(t *testing.T) {
		b := NewProfBuf(2, 11, 1)
		write(t, b, unsafe.Pointer(&myTags[0]), 1, []uint64{2, 3}, []uintptr{4, 5, 6, 7, 8, 9})
		read(t, b, []uint64{10, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []unsafe.Pointer{unsafe.Pointer(&myTags[0])})
		read(t, b, nil, nil) // release data returned by previous read
		write(t, b, unsafe.Pointer(&myTags[2]), 99, []uint64{101, 102}, []uintptr{201, 202, 203, 204})
		read(t, b, []uint64{8, 99, 101, 102, 201, 202, 203, 204}, []unsafe.Pointer{unsafe.Pointer(&myTags[2])})
	})

	t.Run("ReadMany", func(t *testing.T) {
		b := NewProfBuf(2, 50, 50)
		write(t, b, unsafe.Pointer(&myTags[0]), 1, []uint64{2, 3}, []uintptr{4, 5, 6, 7, 8, 9})
		write(t, b, unsafe.Pointer(&myTags[2]), 99, []uint64{101, 102}, []uintptr{201, 202, 203, 204})
		write(t, b, unsafe.Pointer(&myTags[1]), 500, []uint64{502, 504}, []uintptr{506})
		read(t, b, []uint64{10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 99, 101, 102, 201, 202, 203, 204, 5, 500, 502, 504, 506}, []unsafe.Pointer{unsafe.Pointer(&myTags[0]), unsafe.Pointer(&myTags[2]), unsafe.Pointer(&myTags[1])})
	})

	t.Run("ReadManyShortData", func(t *testing.T) {
		b := NewProfBuf(2, 50, 50)
		write(t, b, unsafe.Pointer(&myTags[0]), 1, []uint64{2, 3}, []uintptr{4, 5, 6, 7, 8, 9})
		write(t, b, unsafe.Pointer(&myTags[2]), 99, []uint64{101, 102}, []uintptr{201, 202, 203, 204})
		read(t, b, []uint64{10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 99, 101, 102, 201, 202, 203, 204}, []unsafe.Pointer{unsafe.Pointer(&myTags[0]), unsafe.Pointer(&myTags[2])})
	})

	t.Run("ReadManyShortTags", func(t *testing.T) {
		b := NewProfBuf(2, 50, 50)
		write(t, b, unsafe.Pointer(&myTags[0]), 1, []uint64{2, 3}, []uintptr{4, 5, 6, 7, 8, 9})
		write(t, b, unsafe.Pointer(&myTags[2]), 99, []uint64{101, 102}, []uintptr{201, 202, 203, 204})
		read(t, b, []uint64{10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 99, 101, 102, 201, 202, 203, 204}, []unsafe.Pointer{unsafe.Pointer(&myTags[0]), unsafe.Pointer(&myTags[2])})
	})

	t.Run("ReadAfterOverflow1", func(t *testing.T) {
		// overflow record synthesized by write
		b := NewProfBuf(2, 16, 5)
		write(t, b, unsafe.Pointer(&myTags[0]), 1, []uint64{2, 3}, []uintptr{4, 5, 6, 7, 8, 9})           // uses 10
		read(t, b, []uint64{10, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []unsafe.Pointer{unsafe.Pointer(&myTags[0])}) // reads 10 but still in use until next read
		write(t, b, unsafe.Pointer(&myTags[0]), 1, []uint64{2, 3}, []uintptr{4, 5})                       // uses 6
		read(t, b, []uint64{6, 1, 2, 3, 4, 5}, []unsafe.Pointer{unsafe.Pointer(&myTags[0])})              // reads 6 but still in use until next read
		// now 10 available
		write(t, b, unsafe.Pointer(&myTags[2]), 99, []uint64{101, 102}, []uintptr{201, 202, 203, 204, 205, 206, 207, 208, 209}) // no room
		for i := 0; i < 299; i++ {
			write(t, b, unsafe.Pointer(&myTags[3]), int64(100+i), []uint64{101, 102}, []uintptr{201, 202, 203, 204}) // no room for overflow+this record
		}
		write(t, b, unsafe.Pointer(&myTags[1]), 500, []uint64{502, 504}, []uintptr{506}) // room for overflow+this record
		read(t, b, []uint64{5, 99, 0, 0, 300, 5, 500, 502, 504, 506}, []unsafe.Pointer{nil, unsafe.Pointer(&myTags[1])})
	})

	t.Run("ReadAfterOverflow2", func(t *testing.T) {
		// overflow record synthesized by read
		b := NewProfBuf(2, 16, 5)
		write(t, b, unsafe.Pointer(&myTags[0]), 1, []uint64{2, 3}, []uintptr{4, 5, 6, 7, 8, 9})
		write(t, b, unsafe.Pointer(&myTags[2]), 99, []uint64{101, 102}, []uintptr{201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213})
		for i := 0; i < 299; i++ {
			write(t, b, unsafe.Pointer(&myTags[3]), 100, []uint64{101, 102}, []uintptr{201, 202, 203, 204})
		}
		read(t, b, []uint64{10, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []unsafe.Pointer{unsafe.Pointer(&myTags[0])}) // reads 10 but still in use until next read
		write(t, b, unsafe.Pointer(&myTags[1]), 500, []uint64{502, 504}, []uintptr{})                     // still overflow
		read(t, b, []uint64{5, 99, 0, 0, 301}, []unsafe.Pointer{nil})                                     // overflow synthesized by read
		write(t, b, unsafe.Pointer(&myTags[1]), 500, []uint64{502, 505}, []uintptr{506})                  // written
		read(t, b, []uint64{5, 500, 502, 505, 506}, []unsafe.Pointer{unsafe.Pointer(&myTags[1])})
	})

	t.Run("ReadAtEndAfterOverflow", func(t *testing.T) {
		b := NewProfBuf(2, 12, 5)
		write(t, b, unsafe.Pointer(&myTags[0]), 1, []uint64{2, 3}, []uintptr{4, 5, 6, 7, 8, 9})
		write(t, b, unsafe.Pointer(&myTags[2]), 99, []uint64{101, 102}, []uintptr{201, 202, 203, 204})
		for i := 0; i < 299; i++ {
			write(t, b, unsafe.Pointer(&myTags[3]), 100, []uint64{101, 102}, []uintptr{201, 202, 203, 204})
		}
		read(t, b, []uint64{10, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []unsafe.Pointer{unsafe.Pointer(&myTags[0])})
		read(t, b, []uint64{5, 99, 0, 0, 300}, []unsafe.Pointer{nil})
		write(t, b, unsafe.Pointer(&myTags[1]), 500, []uint64{502, 504}, []uintptr{506})
		read(t, b, []uint64{5, 500, 502, 504, 506}, []unsafe.Pointer{unsafe.Pointer(&myTags[1])})
	})

	t.Run("BlockingWriteRead", func(t *testing.T) {
		b := NewProfBuf(2, 11, 1)
		wait := readBlock(t, b, []uint64{10, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []unsafe.Pointer{unsafe.Pointer(&myTags[0])})
		write(t, b, unsafe.Pointer(&myTags[0]), 1, []uint64{2, 3}, []uintptr{4, 5, 6, 7, 8, 9})
		wait()
		wait = readBlock(t, b, []uint64{8, 99, 101, 102, 201, 202, 203, 204}, []unsafe.Pointer{unsafe.Pointer(&myTags[2])})
		time.Sleep(10 * time.Millisecond)
		write(t, b, unsafe.Pointer(&myTags[2]), 99, []uint64{101, 102}, []uintptr{201, 202, 203, 204})
		wait()
		wait = readBlock(t, b, nil, nil)
		b.Close()
		wait()
		wait = readBlock(t, b, nil, nil)
		wait()
		readEOF(t, b)
	})

	t.Run("DataWraparound", func(t *testing.T) {
		b := NewProfBuf(2, 16, 1024)
		for i := 0; i < 10; i++ {
			write(t, b, unsafe.Pointer(&myTags[0]), 1, []uint64{2, 3}, []uintptr{4, 5, 6, 7, 8, 9})
			read(t, b, []uint64{10, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []unsafe.Pointer{unsafe.Pointer(&myTags[0])})
			read(t, b, nil, nil) // release data returned by previous read
		}
	})

	t.Run("TagWraparound", func(t *testing.T) {
		b := NewProfBuf(2, 1024, 2)
		for i := 0; i < 10; i++ {
			write(t, b, unsafe.Pointer(&myTags[0]), 1, []uint64{2, 3}, []uintptr{4, 5, 6, 7, 8, 9})
			read(t, b, []uint64{10, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []unsafe.Pointer{unsafe.Pointer(&myTags[0])})
			read(t, b, nil, nil) // release data returned by previous read
		}
	})

	t.Run("BothWraparound", func(t *testing.T) {
		b := NewProfBuf(2, 16, 2)
		for i := 0; i < 10; i++ {
			write(t, b, unsafe.Pointer(&myTags[0]), 1, []uint64{2, 3}, []uintptr{4, 5, 6, 7, 8, 9})
			read(t, b, []uint64{10, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []unsafe.Pointer{unsafe.Pointer(&myTags[0])})
			read(t, b, nil, nil) // release data returned by previous read
		}
	})
}

func TestProfBufDoubleWakeup(t *testing.T) {
	b := NewProfBuf(2, 16, 2)
	go func() {
		for range 1000 {
			b.Write(nil, 1, []uint64{5, 6}, []uintptr{7, 8})
		}
		b.Close()
	}()
	for {
		_, _, eof := b.Read(ProfBufBlocking)
		if eof {
			return
		}
	}
}
