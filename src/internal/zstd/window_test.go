// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zstd

import (
	"bytes"
	"fmt"
	"testing"
)

func makeSequence(start, n int) (seq []byte) {
	for i := 0; i < n; i++ {
		seq = append(seq, byte(start+i))
	}
	return
}

func TestWindow(t *testing.T) {
	for size := 0; size <= 3; size++ {
		for i := 0; i <= 2*size; i++ {
			a := makeSequence('a', i)
			for j := 0; j <= 2*size; j++ {
				b := makeSequence('a'+i, j)
				for k := 0; k <= 2*size; k++ {
					c := makeSequence('a'+i+j, k)

					t.Run(fmt.Sprintf("%d-%d-%d-%d", size, i, j, k), func(t *testing.T) {
						testWindow(t, size, a, b, c)
					})
				}
			}
		}
	}
}

// testWindow tests window by saving three sequences of bytes to it.
// Third sequence tests read offset that can become non-zero only after second save.
func testWindow(t *testing.T, size int, a, b, c []byte) {
	var w window
	w.reset(size)

	w.save(a)
	w.save(b)
	w.save(c)

	var tail []byte
	tail = append(tail, a...)
	tail = append(tail, b...)
	tail = append(tail, c...)

	if len(tail) > size {
		tail = tail[len(tail)-size:]
	}

	if w.len() != uint32(len(tail)) {
		t.Errorf("wrong data length: got: %d, want: %d", w.len(), len(tail))
	}

	var from, to uint32
	for from = 0; from <= uint32(len(tail)); from++ {
		for to = from; to <= uint32(len(tail)); to++ {
			got := w.appendTo(nil, from, to)
			want := tail[from:to]

			if !bytes.Equal(got, want) {
				t.Errorf("wrong data at [%d:%d]: got %q, want %q", from, to, got, want)
			}
		}
	}
}
