// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"runtime"
	"testing"
	"time"
)

func TestSystemRoots(t *testing.T) {
	switch runtime.GOARCH {
	case "arm", "arm64":
		t.Skipf("skipping on %s/%s, no system root", runtime.GOOS, runtime.GOARCH)
	}

	switch runtime.GOOS {
	case "darwin":
		t.Skipf("skipping on %s/%s until cgo part of golang.org/issue/16532 has been implemented.", runtime.GOOS, runtime.GOARCH)
	}

	t0 := time.Now()
	sysRoots := systemRootsPool() // actual system roots
	sysRootsDuration := time.Since(t0)

	t1 := time.Now()
	execRoots, err := execSecurityRoots() // non-cgo roots
	execSysRootsDuration := time.Since(t1)

	if err != nil {
		t.Fatalf("failed to read system roots: %v", err)
	}

	t.Logf("    cgo sys roots: %v", sysRootsDuration)
	t.Logf("non-cgo sys roots: %v", execSysRootsDuration)

	for _, tt := range []*CertPool{sysRoots, execRoots} {
		if tt == nil {
			t.Fatal("no system roots")
		}
		// On Mavericks, there are 212 bundled certs, at least
		// there was at one point in time on one machine.
		// (Maybe it was a corp laptop with extra certs?)
		// Other OS X users report
		// 135, 142, 145...  Let's try requiring at least 100,
		// since this is just a sanity check.
		t.Logf("got %d roots", len(tt.certs))
		if want, have := 100, len(tt.certs); have < want {
			t.Fatalf("want at least %d system roots, have %d", want, have)
		}
	}

	// Check that the two cert pools are roughly the same;
	// |A∩B| > max(|A|, |B|) / 2 should be a reasonably robust check.

	isect := make(map[string]bool, len(sysRoots.certs))
	for _, c := range sysRoots.certs {
		isect[string(c.Raw)] = true
	}

	have := 0
	for _, c := range execRoots.certs {
		if isect[string(c.Raw)] {
			have++
		}
	}

	var want int
	if nsys, nexec := len(sysRoots.certs), len(execRoots.certs); nsys > nexec {
		want = nsys / 2
	} else {
		want = nexec / 2
	}

	if have < want {
		t.Errorf("insufficient overlap between cgo and non-cgo roots; want at least %d, have %d", want, have)
	}
}
