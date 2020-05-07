// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows cryptographically secure pseudorandom number
// generator.

package rand

import (
	"internal/syscall/windows"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// Implemented by using Windows CryptoAPI 2.0.

func init() { Reader = &rngReader{} }

// A rngReader satisfies reads by reading from the Windows CryptGenRandom API.
type rngReader struct {
	used int32 // atomic; whether this rngReader has been used
	prov syscall.Handle
	mu   sync.Mutex
}

func (r *rngReader) Read(b []byte) (n int, err error) {
	if atomic.CompareAndSwapInt32(&r.used, 0, 1) {
		// First use of randomness. Start timer to warn about
		// being blocked on entropy not being available.
		t := time.AfterFunc(60*time.Second, warnBlocked)
		defer t.Stop()
	}
	r.mu.Lock()
	if r.prov == 0 {
		// BCRYPT_RNG_ALGORITHM is defined here:
		// https://docs.microsoft.com/en-us/windows/win32/seccng/cng-algorithm-identifiers
		// Standard: FIPS 186-2, FIPS 140-2, NIST SP 800-90
		algID, err := syscall.UTF16PtrFromString(windows.BCRYPT_RNG_ALGORITHM)
		if err != nil {
			r.mu.Unlock()
			return 0, err
		}
		status := windows.BCryptOpenAlgorithmProvider(&r.prov, algID, nil, 0)
		if status != 0 {
			r.mu.Unlock()
			return 0, os.NewSyscallError("BCryptOpenAlgorithmProvider", syscall.Errno(status))
		}
	}
	r.mu.Unlock()

	if len(b) == 0 {
		return 0, nil
	}
	status := windows.BCryptGenRandom(r.prov, &b[0], uint32(len(b)), 0)
	if status != 0 {
		return 0, os.NewSyscallError("BCryptGenRandom", syscall.Errno(status))
	}
	return int(uint32(len(b))), nil
}
