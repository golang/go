// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zstd

// window stores up to size bytes of data.
// It is implemented as a circular buffer:
// sequential save calls append to the data slice until
// its length reaches configured size and after that,
// save calls overwrite previously saved data at off
// and update off such that it always points at
// the byte stored before others.
type window struct {
	size int
	data []byte
	off  int
}

// reset clears stored data and configures window size.
func (w *window) reset(size int) {
	w.data = w.data[:0]
	w.off = 0
	w.size = size
}

// len returns the number of stored bytes.
func (w *window) len() uint32 {
	return uint32(len(w.data))
}

// save stores up to size last bytes from the buf.
func (w *window) save(buf []byte) {
	if w.size == 0 {
		return
	}
	if len(buf) == 0 {
		return
	}

	if len(buf) >= w.size {
		from := len(buf) - w.size
		w.data = append(w.data[:0], buf[from:]...)
		w.off = 0
		return
	}

	// Update off to point to the oldest remaining byte.
	free := w.size - len(w.data)
	if free == 0 {
		n := copy(w.data[w.off:], buf)
		if n == len(buf) {
			w.off += n
		} else {
			w.off = copy(w.data, buf[n:])
		}
	} else {
		if free >= len(buf) {
			w.data = append(w.data, buf...)
		} else {
			w.data = append(w.data, buf[:free]...)
			w.off = copy(w.data, buf[free:])
		}
	}
}

// appendTo appends stored bytes between from and to indices to the buf.
// Index from must be less or equal to index to and to must be less or equal to w.len().
func (w *window) appendTo(buf []byte, from, to uint32) []byte {
	dataLen := uint32(len(w.data))
	from += uint32(w.off)
	to += uint32(w.off)

	wrap := false
	if from > dataLen {
		from -= dataLen
		wrap = !wrap
	}
	if to > dataLen {
		to -= dataLen
		wrap = !wrap
	}

	if wrap {
		buf = append(buf, w.data[from:]...)
		return append(buf, w.data[:to]...)
	} else {
		return append(buf, w.data[from:to]...)
	}
}
