// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux
// +build linux

package net

import (
	"io"
	"os"
	"strconv"
	"testing"
)

func BenchmarkSendFile(b *testing.B) {
	for i := 0; i <= 10; i++ {
		size := 1 << (i + 10)
		bench := sendFileBench{chunkSize: size}
		b.Run(strconv.Itoa(size), bench.benchSendFile)
	}
}

type sendFileBench struct {
	chunkSize int
}

func (bench sendFileBench) benchSendFile(b *testing.B) {
	fileSize := b.N * bench.chunkSize
	f := createTempFile(b, fileSize)
	fileName := f.Name()
	defer os.Remove(fileName)
	defer f.Close()

	client, server := spliceTestSocketPair(b, "tcp")
	defer server.Close()

	cleanUp, err := startSpliceClient(client, "r", bench.chunkSize, fileSize)
	if err != nil {
		b.Fatal(err)
	}
	defer cleanUp()

	b.ReportAllocs()
	b.SetBytes(int64(bench.chunkSize))
	b.ResetTimer()

	// Data go from file to socket via sendfile(2).
	sent, err := io.Copy(server, f)
	if err != nil {
		b.Fatalf("failed to copy data with sendfile, error: %v", err)
	}
	if sent != int64(fileSize) {
		b.Fatalf("bytes sent mismatch\n\texpect: %d\n\tgot: %d", fileSize, sent)
	}
}

func createTempFile(b *testing.B, size int) *os.File {
	f, err := os.CreateTemp("", "linux-sendfile-test")
	if err != nil {
		b.Fatalf("failed to create tmp directory: %v", err)
	}

	data := make([]byte, size)
	if _, err := f.Write(data); err != nil {
		b.Fatalf("failed to create and feed the file: %v", err)
	}
	if err := f.Sync(); err != nil {
		b.Fatalf("failed to save the file: %v", err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		b.Fatalf("failed to rewind the file: %v", err)
	}

	return f
}
