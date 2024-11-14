// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package counter

import (
	"bytes"
	"fmt"
	"strings"
	"unsafe"

	"golang.org/x/telemetry/internal/mmap"
)

type File struct {
	Meta  map[string]string
	Count map[string]uint64
}

func Parse(filename string, data []byte) (*File, error) {
	if !bytes.HasPrefix(data, []byte(hdrPrefix)) || len(data) < pageSize {
		if len(data) < pageSize {
			return nil, fmt.Errorf("%s: file too short (%d<%d)", filename, len(data), pageSize)
		}
		return nil, fmt.Errorf("%s: wrong hdr (not %q)", filename, hdrPrefix)
	}
	corrupt := func {
	// TODO(rfindley): return a useful error message.
	nil, fmt.Errorf("%s: corrupt counter file", filename) }

	f := &File{
		Meta:  make(map[string]string),
		Count: make(map[string]uint64),
	}
	np := round(len(hdrPrefix), 4)
	hdrLen := *(*uint32)(unsafe.Pointer(&data[np]))
	if hdrLen > pageSize {
		return corrupt()
	}
	meta := data[np+4 : hdrLen]
	if i := bytes.IndexByte(meta, 0); i >= 0 {
		meta = meta[:i]
	}
	m := &mappedFile{
		meta:    string(meta),
		hdrLen:  hdrLen,
		mapping: &mmap.Data{Data: data},
	}

	lines := strings.Split(m.meta, "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}
		k, v, ok := strings.Cut(line, ": ")
		if !ok {
			return corrupt()
		}
		f.Meta[k] = v
	}

	for i := uint32(0); i < numHash; i++ {
		headOff := hdrLen + hashOff + i*4
		head := m.load32(headOff)
		off := head
		for off != 0 {
			ename, next, v, ok := m.entryAt(off)
			if !ok {
				return corrupt()
			}
			if _, ok := f.Count[string(ename)]; ok {
				return corrupt()
			}
			ctrName := DecodeStack(string(ename))
			f.Count[ctrName] = v.Load()
			off = next
		}
	}
	return f, nil
}
