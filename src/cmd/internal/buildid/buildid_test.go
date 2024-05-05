// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildid

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"internal/binary"
	"internal/obscuretestdata"
	"os"
	"reflect"
	"strings"
	"testing"
)

const (
	expectedID = "abcdefghijklmnopqrstuvwxyz.1234567890123456789012345678901234567890123456789012345678901234"
	newID      = "bcdefghijklmnopqrstuvwxyza.2345678901234567890123456789012345678901234567890123456789012341"
)

func TestReadFile(t *testing.T) {
	f, err := os.CreateTemp("", "buildid-test-")
	if err != nil {
		t.Fatal(err)
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	f.Close()

	// Use obscured files to prevent Apple’s notarization service from
	// mistaking them as candidates for notarization and rejecting the entire
	// toolchain.
	// See golang.org/issue/34986
	var files = []string{
		"p.a.base64",
		"a.elf.base64",
		"a.macho.base64",
		"a.pe.base64",
	}

	for _, name := range files {
		f, err := obscuretestdata.DecodeToTempFile("testdata/" + name)
		if err != nil {
			t.Errorf("obscuretestdata.DecodeToTempFile(testdata/%s): %v", name, err)
			continue
		}
		defer os.Remove(f)
		id, err := ReadFile(f)
		if id != expectedID || err != nil {
			t.Errorf("ReadFile(testdata/%s) = %q, %v, want %q, nil", f, id, err, expectedID)
		}
		old := readSize
		readSize = 2048
		id, err = ReadFile(f)
		readSize = old
		if id != expectedID || err != nil {
			t.Errorf("ReadFile(%s) [readSize=2k] = %q, %v, want %q, nil", f, id, err, expectedID)
		}

		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		m, _, err := FindAndHash(bytes.NewReader(data), expectedID, 1024)
		if err != nil {
			t.Errorf("FindAndHash(%s): %v", f, err)
			continue
		}
		if err := os.WriteFile(tmp, data, 0666); err != nil {
			t.Error(err)
			continue
		}
		tf, err := os.OpenFile(tmp, os.O_WRONLY, 0)
		if err != nil {
			t.Error(err)
			continue
		}
		err = Rewrite(tf, m, newID)
		err2 := tf.Close()
		if err != nil {
			t.Errorf("Rewrite(%s): %v", f, err)
			continue
		}
		if err2 != nil {
			t.Fatal(err2)
		}

		id, err = ReadFile(tmp)
		if id != newID || err != nil {
			t.Errorf("ReadFile(%s after Rewrite) = %q, %v, want %q, nil", f, id, err, newID)
		}

		// Test an ELF PT_NOTE segment with an Align field of 0.
		// Do this by rewriting the file data.
		if strings.Contains(name, "elf") {
			// We only expect a 64-bit ELF file.
			if elf.Class(data[elf.EI_CLASS]) != elf.ELFCLASS64 {
				continue
			}

			// We only expect a little-endian ELF file.
			if elf.Data(data[elf.EI_DATA]) != elf.ELFDATA2LSB {
				continue
			}
			order := binary.LittleEndian

			var hdr elf.Header64
			if err := binary.Read(bytes.NewReader(data), order, &hdr); err != nil {
				t.Error(err)
				continue
			}

			phoff := hdr.Phoff
			phnum := int(hdr.Phnum)
			phsize := uint64(hdr.Phentsize)

			for i := 0; i < phnum; i++ {
				var phdr elf.Prog64
				if err := binary.Read(bytes.NewReader(data[phoff:]), order, &phdr); err != nil {
					t.Error(err)
					continue
				}

				if elf.ProgType(phdr.Type) == elf.PT_NOTE {
					// Increase the size so we keep
					// reading notes.
					order.PutUint64(data[phoff+4*8:], phdr.Filesz+1)

					// Clobber the Align field to zero.
					order.PutUint64(data[phoff+6*8:], 0)

					// Clobber the note type so we
					// keep reading notes.
					order.PutUint32(data[phdr.Off+12:], 0)
				}

				phoff += phsize
			}

			if err := os.WriteFile(tmp, data, 0666); err != nil {
				t.Error(err)
				continue
			}

			id, err := ReadFile(tmp)
			// Because we clobbered the note type above,
			// we don't expect to see a Go build ID.
			// The issue we are testing for was a crash
			// in Readfile; see issue #62097.
			if id != "" || err != nil {
				t.Errorf("ReadFile with zero ELF Align = %q, %v, want %q, nil", id, err, "")
				continue
			}
		}
	}
}

func TestFindAndHash(t *testing.T) {
	buf := make([]byte, 64)
	buf2 := make([]byte, 64)
	id := make([]byte, 8)
	zero := make([]byte, 8)
	for i := range id {
		id[i] = byte(i)
	}
	numError := 0
	errorf := func(msg string, args ...any) {
		t.Errorf(msg, args...)
		if numError++; numError > 20 {
			t.Logf("stopping after too many errors")
			t.FailNow()
		}
	}
	for bufSize := len(id); bufSize <= len(buf); bufSize++ {
		for j := range buf {
			for k := 0; k < 2*len(id) && j+k < len(buf); k++ {
				for i := range buf {
					buf[i] = 1
				}
				copy(buf[j:], id)
				copy(buf[j+k:], id)
				var m []int64
				if j+len(id) <= j+k {
					m = append(m, int64(j))
				}
				if j+k+len(id) <= len(buf) {
					m = append(m, int64(j+k))
				}
				copy(buf2, buf)
				for _, p := range m {
					copy(buf2[p:], zero)
				}
				h := sha256.Sum256(buf2)

				matches, hash, err := FindAndHash(bytes.NewReader(buf), string(id), bufSize)
				if err != nil {
					errorf("bufSize=%d j=%d k=%d: findAndHash: %v", bufSize, j, k, err)
					continue
				}
				if !reflect.DeepEqual(matches, m) {
					errorf("bufSize=%d j=%d k=%d: findAndHash: matches=%v, want %v", bufSize, j, k, matches, m)
					continue
				}
				if hash != h {
					errorf("bufSize=%d j=%d k=%d: findAndHash: matches correct, but hash=%x, want %x", bufSize, j, k, hash, h)
				}
			}
		}
	}
}

func TestExcludedReader(t *testing.T) {
	const s = "0123456789abcdefghijklmn"
	tests := []struct {
		start, end int64    // excluded range
		results    []string // expected results of reads
	}{
		{12, 15, []string{"0123456789", "ab\x00\x00\x00fghij", "klmn"}},                              // within one read
		{8, 21, []string{"01234567\x00\x00", "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", "\x00lmn"}}, // across multiple reads
		{10, 20, []string{"0123456789", "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", "klmn"}},         // a whole read
		{0, 5, []string{"\x00\x00\x00\x00\x0056789", "abcdefghij", "klmn"}},                          // start
		{12, 24, []string{"0123456789", "ab\x00\x00\x00\x00\x00\x00\x00\x00", "\x00\x00\x00\x00"}},   // end
	}
	p := make([]byte, 10)
	for _, test := range tests {
		r := &excludedReader{strings.NewReader(s), 0, test.start, test.end}
		for _, res := range test.results {
			n, err := r.Read(p)
			if err != nil {
				t.Errorf("read failed: %v", err)
			}
			if n != len(res) {
				t.Errorf("unexpected number of bytes read: want %d, got %d", len(res), n)
			}
			if string(p[:n]) != res {
				t.Errorf("unexpected bytes: want %q, got %q", res, p[:n])
			}
		}
	}
}

func TestEmptyID(t *testing.T) {
	r := strings.NewReader("aha!")
	matches, hash, err := FindAndHash(r, "", 1000)
	if matches != nil || hash != ([32]byte{}) || err == nil || !strings.Contains(err.Error(), "no id") {
		t.Errorf("FindAndHash: want nil, [32]byte{}, no id specified, got %v, %v, %v", matches, hash, err)
	}
}
