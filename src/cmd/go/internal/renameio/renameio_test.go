// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9

package renameio

import (
	"encoding/binary"
	"errors"
	"internal/testenv"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"cmd/go/internal/robustio"
)

func TestConcurrentReadsAndWrites(t *testing.T) {
	if runtime.GOOS == "darwin" && strings.HasSuffix(testenv.Builder(), "-10_14") {
		testenv.SkipFlaky(t, 33041)
	}

	dir, err := os.MkdirTemp("", "renameio")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "blob.bin")

	const chunkWords = 8 << 10
	buf := make([]byte, 2*chunkWords*8)
	for i := uint64(0); i < 2*chunkWords; i++ {
		binary.LittleEndian.PutUint64(buf[i*8:], i)
	}

	var attempts int64 = 128
	if !testing.Short() {
		attempts *= 16
	}
	const parallel = 32

	var sem = make(chan bool, parallel)

	var (
		writeSuccesses, readSuccesses int64 // atomic
		writeErrnoSeen, readErrnoSeen sync.Map
	)

	for n := attempts; n > 0; n-- {
		sem <- true
		go func() {
			defer func() { <-sem }()

			time.Sleep(time.Duration(rand.Intn(100)) * time.Microsecond)
			offset := rand.Intn(chunkWords)
			chunk := buf[offset*8 : (offset+chunkWords)*8]
			if err := WriteFile(path, chunk, 0666); err == nil {
				atomic.AddInt64(&writeSuccesses, 1)
			} else if robustio.IsEphemeralError(err) {
				var (
					errno syscall.Errno
					dup   bool
				)
				if errors.As(err, &errno) {
					_, dup = writeErrnoSeen.LoadOrStore(errno, true)
				}
				if !dup {
					t.Logf("ephemeral error: %v", err)
				}
			} else {
				t.Errorf("unexpected error: %v", err)
			}

			time.Sleep(time.Duration(rand.Intn(100)) * time.Microsecond)
			data, err := ReadFile(path)
			if err == nil {
				atomic.AddInt64(&readSuccesses, 1)
			} else if robustio.IsEphemeralError(err) {
				var (
					errno syscall.Errno
					dup   bool
				)
				if errors.As(err, &errno) {
					_, dup = readErrnoSeen.LoadOrStore(errno, true)
				}
				if !dup {
					t.Logf("ephemeral error: %v", err)
				}
				return
			} else {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if len(data) != 8*chunkWords {
				t.Errorf("read %d bytes, but each write is a %d-byte file", len(data), 8*chunkWords)
				return
			}

			u := binary.LittleEndian.Uint64(data)
			for i := 1; i < chunkWords; i++ {
				next := binary.LittleEndian.Uint64(data[i*8:])
				if next != u+1 {
					t.Errorf("wrote sequential integers, but read integer out of sequence at offset %d", i)
					return
				}
				u = next
			}
		}()
	}

	for n := parallel; n > 0; n-- {
		sem <- true
	}

	var minWriteSuccesses int64 = attempts
	if runtime.GOOS == "windows" {
		// Windows produces frequent "Access is denied" errors under heavy rename load.
		// As long as those are the only errors and *some* of the writes succeed, we're happy.
		minWriteSuccesses = attempts / 4
	}

	if writeSuccesses < minWriteSuccesses {
		t.Errorf("%d (of %d) writes succeeded; want ≥ %d", writeSuccesses, attempts, minWriteSuccesses)
	} else {
		t.Logf("%d (of %d) writes succeeded (ok: ≥ %d)", writeSuccesses, attempts, minWriteSuccesses)
	}

	var minReadSuccesses int64 = attempts

	switch runtime.GOOS {
	case "windows":
		// Windows produces frequent "Access is denied" errors under heavy rename load.
		// As long as those are the only errors and *some* of the reads succeed, we're happy.
		minReadSuccesses = attempts / 4

	case "darwin", "ios":
		// The filesystem on certain versions of macOS (10.14) and iOS (affected
		// versions TBD) occasionally fail with "no such file or directory" errors.
		// See https://golang.org/issue/33041 and https://golang.org/issue/42066.
		// The flake rate is fairly low, so ensure that at least 75% of attempts
		// succeed.
		minReadSuccesses = attempts - (attempts / 4)
	}

	if readSuccesses < minReadSuccesses {
		t.Errorf("%d (of %d) reads succeeded; want ≥ %d", readSuccesses, attempts, minReadSuccesses)
	} else {
		t.Logf("%d (of %d) reads succeeded (ok: ≥ %d)", readSuccesses, attempts, minReadSuccesses)
	}
}
