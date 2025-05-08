// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"encoding/pem"
	"fmt"
	"runtime"
	"testing"
	"time"
)

func TestCertCache(t *testing.T) {
	cc := certCache{}
	p, _ := pem.Decode([]byte(rsaCertPEM))
	if p == nil {
		t.Fatal("Failed to decode certificate")
	}

	certA, err := cc.newCert(p.Bytes)
	if err != nil {
		t.Fatalf("newCert failed: %s", err)
	}
	certB, err := cc.newCert(p.Bytes)
	if err != nil {
		t.Fatalf("newCert failed: %s", err)
	}
	if certA.cert != certB.cert {
		t.Fatal("newCert returned a unique reference for a duplicate certificate")
	}

	if entry, ok := cc.Load(string(p.Bytes)); !ok {
		t.Fatal("cache does not contain expected entry")
	} else {
		if refs := entry.(*cacheEntry).refs.Load(); refs != 2 {
			t.Fatalf("unexpected number of references: got %d, want 2", refs)
		}
	}

	timeoutRefCheck := func(t *testing.T, key string, count int64) {
		t.Helper()
		timeout := time.After(4 * time.Second)
		for {
			select {
			case <-timeout:
				t.Fatal("timed out waiting for expected ref count")
			default:
				e, ok := cc.Load(key)
				if !ok && count != 0 {
					t.Fatal("cache does not contain expected key")
				} else if count == 0 && !ok {
					return
				}

				if e.(*cacheEntry).refs.Load() == count {
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

	timeoutRefCheck(t, string(p.Bytes), 1)

	// Keep certB alive until at least now, so that we can
	// purposefully nil it and force the finalizer to be
	// called.
	runtime.KeepAlive(certB)
	certB = nil
	runtime.GC()

	timeoutRefCheck(t, string(p.Bytes), 0)
}

func BenchmarkCertCache(b *testing.B) {
	p, _ := pem.Decode([]byte(rsaCertPEM))
	if p == nil {
		b.Fatal("Failed to decode certificate")
	}

	cc := certCache{}
	b.ReportAllocs()
	b.ResetTimer()
	// We expect that calling newCert additional times after
	// the initial call should not cause additional allocations.
	for extra := 0; extra < 4; extra++ {
		b.Run(fmt.Sprint(extra), func(b *testing.B) {
			actives := make([]*activeCert, extra+1)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var err error
				actives[0], err = cc.newCert(p.Bytes)
				if err != nil {
					b.Fatal(err)
				}
				for j := 0; j < extra; j++ {
					actives[j+1], err = cc.newCert(p.Bytes)
					if err != nil {
						b.Fatal(err)
					}
				}
				for j := 0; j < extra+1; j++ {
					actives[j] = nil
				}
				runtime.GC()
			}
		})
	}
}
