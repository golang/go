// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || solaris

package os_test

import (
	"io"
	. "os"
	"testing"
)

func BenchmarkSendFile(b *testing.B) {
	hook := hookSendFileTB(b)

	// 1 GiB file size for copy.
	const fileSize = 1 << 30

	src, _ := createTempFile(b, "benchmark-sendfile-src", int64(fileSize))
	dst, err := CreateTemp(b.TempDir(), "benchmark-sendfile-dst")
	if err != nil {
		b.Fatalf("failed to create temporary file of destination: %v", err)
	}
	b.Cleanup(func() {
		dst.Close()
	})

	b.ReportAllocs()
	b.SetBytes(int64(fileSize))
	b.ResetTimer()

	for i := 0; i <= b.N; i++ {
		sent, err := io.Copy(dst, src)

		if err != nil {
			b.Fatalf("failed to copy data: %v", err)
		}
		if !hook.called {
			b.Fatalf("should have called the sendfile(2)")
		}
		if sent != int64(fileSize) {
			b.Fatalf("sent %d bytes, want %d", sent, fileSize)
		}

		// Rewind the files for the next iteration.
		if _, err := src.Seek(0, io.SeekStart); err != nil {
			b.Fatalf("failed to rewind the source file: %v", err)
		}
		if _, err := dst.Seek(0, io.SeekStart); err != nil {
			b.Fatalf("failed to rewind the destination file: %v", err)
		}
	}
}
