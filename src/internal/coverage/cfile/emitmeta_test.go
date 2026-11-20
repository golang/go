// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfile

import (
	"encoding/binary"
	"fmt"
	"internal/coverage"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"unsafe"
)

// writeMetaHeaderFile writes a meta-data file containing only a header
// for metaHash, with TotalLength matching the file size, and returns
// its path. Extra bytes can be appended afterwards to simulate a
// corrupt or partial file.
func writeMetaHeaderFile(t *testing.T, dir string, metaHash [16]byte) string {
	t.Helper()
	fn := fmt.Sprintf("%s.%x", coverage.MetaFilePref, metaHash)
	path := filepath.Join(dir, fn)
	mh := coverage.MetaFileHeader{
		Magic:        coverage.CovMetaMagic,
		Version:      coverage.MetaFileVersion,
		TotalLength:  uint64(unsafe.Sizeof(coverage.MetaFileHeader{})),
		MetaFileHash: metaHash,
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(f, binary.LittleEndian, &mh); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestMetaFileReusable(t *testing.T) {
	var hash [16]byte
	copy(hash[:], "0123456789abcdef")
	dir := t.TempDir()

	if metaFileReusable(filepath.Join(dir, "covmeta.doesnotexist"), hash) {
		t.Error("metaFileReusable(missing) = true, want false")
	}

	garbage := filepath.Join(dir, "covmeta.garbage")
	if err := os.WriteFile(garbage, []byte("not a meta file"), 0666); err != nil {
		t.Fatal(err)
	}
	if metaFileReusable(garbage, hash) {
		t.Error("metaFileReusable(garbage) = true, want false")
	}

	path := writeMetaHeaderFile(t, dir, hash)
	if !metaFileReusable(path, hash) {
		t.Error("metaFileReusable(valid) = false, want true")
	}

	var other [16]byte
	copy(other[:], "fedcba9876543210")
	if metaFileReusable(path, other) {
		t.Error("metaFileReusable(wrong hash) = true, want false")
	}

	// Corrupt the file so its size no longer matches TotalLength.
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write([]byte{0}); err != nil {
		t.Fatal(err)
	}
	f.Close()
	if metaFileReusable(path, hash) {
		t.Error("metaFileReusable(length mismatch) = true, want false")
	}
}

func TestOpenMetaFileReusesCompleteFile(t *testing.T) {
	var hash [16]byte
	copy(hash[:], "0123456789abcdef")
	dir := t.TempDir()
	writeMetaHeaderFile(t, dir, hash)

	s := &emitState{outdir: dir}
	if err := s.openMetaFile(hash); err != nil {
		t.Fatal(err)
	}
	if s.mf != nil || s.mftmp != "" {
		t.Errorf("openMetaFile opened a new temp file for a reusable meta file (mftmp=%q)", s.mftmp)
		if s.mf != nil {
			s.mf.Close()
		}
	}
}

func TestOpenMetaFileTempNameHasPid(t *testing.T) {
	var hash [16]byte
	copy(hash[:], "0123456789abcdef")
	dir := t.TempDir()

	// The temp name must carry the pid: without it, concurrent
	// processes sharing a GOCOVERDIR can pick the same temp name
	// and all but one fail to rename (issue #82911).
	s := &emitState{outdir: dir}
	if err := s.openMetaFile(hash); err != nil {
		t.Fatal(err)
	}
	if s.mf == nil {
		t.Fatal("openMetaFile did not open a temp file in an empty directory")
	}
	defer s.mf.Close()
	want := "." + strconv.Itoa(os.Getpid()) + "."
	if got := s.mftmp; !strings.Contains(got, want) {
		t.Errorf("temp meta name %q does not contain pid fragment %q", got, want)
	}
}
