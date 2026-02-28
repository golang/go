// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package tls

import (
	"encoding/pem"
	"runtime"
	"testing"
	"time"
)

func TestWeakCertCache(t *testing.T) {
	wcc := weakCertCache{}
	p, _ := pem.Decode([]byte(rsaCertPEM))
	if p == nil {
		t.Fatal("Failed to decode certificate")
	}

	certA, err := wcc.newCert(p.Bytes)
	if err != nil {
		t.Fatalf("newCert failed: %s", err)
	}
	certB, err := wcc.newCert(p.Bytes)
	if err != nil {
		t.Fatalf("newCert failed: %s", err)
	}
	if certA != certB {
		t.Fatal("newCert returned a unique reference for a duplicate certificate")
	}

	if _, ok := wcc.Load(string(p.Bytes)); !ok {
		t.Fatal("cache does not contain expected entry")
	}

	timeoutRefCheck := func(t *testing.T, key string, present bool) {
		t.Helper()
		timeout := time.After(4 * time.Second)
		for {
			select {
			case <-timeout:
				t.Fatal("timed out waiting for expected ref count")
			default:
				_, ok := wcc.Load(key)
				if ok == present {
					return
				}
			}
			// Explicitly yield to the scheduler.
			//
			// On single-threaded platforms like js/wasm a busy-loop might
			// never call into the scheduler for the full timeout, meaning
			// that if we arrive here and the cleanup hasn't already run,
			// we'll simply loop until the timeout. Busy-loops put us at the
			// mercy of the Go scheduler, making this test fragile on some
			// platforms.
			runtime.Gosched()
		}
	}

	// Keep certA alive until at least now, so that we can
	// purposefully nil it and force the finalizer to be
	// called.
	runtime.KeepAlive(certA)
	certA = nil
	runtime.GC()

	timeoutRefCheck(t, string(p.Bytes), true)

	// Keep certB alive until at least now, so that we can
	// purposefully nil it and force the finalizer to be
	// called.
	runtime.KeepAlive(certB)
	certB = nil
	runtime.GC()

	timeoutRefCheck(t, string(p.Bytes), false)
}
