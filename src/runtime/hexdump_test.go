// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"internal/abi"
	"internal/goarch"
	"runtime"
	"slices"
	"strings"
	"testing"
	"unsafe"
)

func TestHexdumper(t *testing.T) {
	check := func(label, got, want string) {
		got = strings.TrimRight(got, "\n")
		want = strings.TrimPrefix(want, "\n")
		want = strings.TrimRight(want, "\n")
		if got != want {
			t.Errorf("%s: got\n%s\nwant\n%s", label, got, want)
		}
	}

	data := make([]byte, 32)
	for i := range data {
		data[i] = 0x10 + byte(i)
	}

	check("basic", runtime.Hexdumper(0, 1, nil, data), `
           0 1 2 3  4 5 6 7   8 9 a b  c d e f  0123456789abcdef
00000000: 10111213 14151617  18191a1b 1c1d1e1f  ................
00000010: 20212223 24252627  28292a2b 2c2d2e2f   !"#$%&'()*+,-./`)

	if !goarch.BigEndian {
		// Different word sizes
		check("word=4", runtime.Hexdumper(0, 4, nil, data), `
           3 2 1 0  7 6 5 4   b a 9 8  f e d c  0123456789abcdef
00000000: 13121110 17161514  1b1a1918 1f1e1d1c  ................
00000010: 23222120 27262524  2b2a2928 2f2e2d2c   !"#$%&'()*+,-./`)
		check("word=8", runtime.Hexdumper(0, 8, nil, data), `
           7 6 5 4  3 2 1 0   f e d c  b a 9 8  0123456789abcdef
00000000: 17161514 13121110  1f1e1d1c 1b1a1918  ................
00000010: 27262524 23222120  2f2e2d2c 2b2a2928   !"#$%&'()*+,-./`)
	}

	// Starting offset
	check("offset=1", runtime.Hexdumper(1, 1, nil, data), `
           0 1 2 3  4 5 6 7   8 9 a b  c d e f  0123456789abcdef
00000000:   101112 13141516  1718191a 1b1c1d1e   ...............
00000010: 1f202122 23242526  2728292a 2b2c2d2e  . !"#$%&'()*+,-.
00000020: 2f                                    /`)
	if !goarch.BigEndian {
		// ... combined with a word size
		check("offset=1 and word=4", runtime.Hexdumper(1, 4, nil, data), `
           3 2 1 0  7 6 5 4   b a 9 8  f e d c  0123456789abcdef
00000000: 121110   16151413  1a191817 1e1d1c1b   ...............
00000010: 2221201f 26252423  2a292827 2e2d2c2b  . !"#$%&'()*+,-.
00000020:       2f                              /`)
	}

	// Partial data full of annoying boundaries.
	partials := make([][]byte, 0)
	for i := 0; i < len(data); i += 2 {
		partials = append(partials, data[i:i+2])
	}
	check("partials", runtime.Hexdumper(1, 1, nil, partials...), `
           0 1 2 3  4 5 6 7   8 9 a b  c d e f  0123456789abcdef
00000000:   101112 13141516  1718191a 1b1c1d1e   ...............
00000010: 1f202122 23242526  2728292a 2b2c2d2e  . !"#$%&'()*+,-.
00000020: 2f                                    /`)

	// Marks.
	check("marks", runtime.Hexdumper(0, 1, func(addr uintptr, start func()) {
		if addr%7 == 0 {
			start()
			println("mark")
		}
	}, data), `
           0 1 2 3  4 5 6 7   8 9 a b  c d e f  0123456789abcdef
00000000: 10111213 14151617  18191a1b 1c1d1e1f  ................
          ^ mark
                         ^ mark
                                          ^ mark
00000010: 20212223 24252627  28292a2b 2c2d2e2f   !"#$%&'()*+,-./
                     ^ mark
                                      ^ mark`)
	if !goarch.BigEndian {
		check("marks and word=4", runtime.Hexdumper(0, 4, func(addr uintptr, start func()) {
			if addr%7 == 0 {
				start()
				println("mark")
			}
		}, data), `
           3 2 1 0  7 6 5 4   b a 9 8  f e d c  0123456789abcdef
00000000: 13121110 17161514  1b1a1918 1f1e1d1c  ................
          ^ mark
00000010: 23222120 27262524  2b2a2928 2f2e2d2c   !"#$%&'()*+,-./
                                      ^ mark`)
	}
}

func TestHexdumpWords(t *testing.T) {
	if goarch.BigEndian || goarch.PtrSize != 8 {
		// We could support these, but it's kind of a pain.
		t.Skip("requires 64-bit little endian")
	}

	// Most of this is in hexdumper. Here we just test the symbolizer.

	pc := abi.FuncPCABIInternal(TestHexdumpWords)
	pcs := slices.Repeat([]uintptr{pc}, 3)

	// Make sure pcs doesn't move around on us.
	var p runtime.Pinner
	defer p.Unpin()
	p.Pin(&pcs[0])
	// Get a 16 byte, 16-byte-aligned chunk of pcs so the hexdump is simple.
	start := uintptr(unsafe.Pointer(&pcs[0]))
	start = (start + 15) &^ uintptr(15)

	// Do the hex dump.
	got := runtime.HexdumpWords(start, 16)

	// Construct the expected output.
	pcStr := fmt.Sprintf("%016x", pc)
	pcStr = pcStr[:8] + " " + pcStr[8:] // Add middle space
	ascii := make([]byte, 8)
	for i := range ascii {
		b := byte(pc >> (8 * i))
		if b >= ' ' && b <= '~' {
			ascii[i] = b
		} else {
			ascii[i] = '.'
		}
	}
	want := fmt.Sprintf(`
                   7 6 5 4  3 2 1 0   f e d c  b a 9 8  0123456789abcdef
%016x: %s  %s  %s%s
                  ^ <runtime_test.TestHexdumpWords+0x0>
                                     ^ <runtime_test.TestHexdumpWords+0x0>
`, start, pcStr, pcStr, ascii, ascii)
	want = strings.TrimPrefix(want, "\n")

	if got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
}
