// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"crypto/rsa"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestSystemRoots(t *testing.T) {
	switch runtime.GOARCH {
	case "arm", "arm64":
		t.Skipf("skipping on %s/%s, no system root", runtime.GOOS, runtime.GOARCH)
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

	// On Mavericks, there are 212 bundled certs, at least there was at
	// one point in time on one machine. (Maybe it was a corp laptop
	// with extra certs?) Other OS X users report 135, 142, 145...
	// Let's try requiring at least 100, since this is just a sanity
	// check.
	if want, have := 100, len(sysRoots.certs); have < want {
		t.Errorf("want at least %d system roots, have %d", want, have)
	}

	// Fetch any intermediate certificate that verify-cert might be aware of.
	out, err := exec.Command("/usr/bin/security", "find-certificate", "-a", "-p",
		"/Library/Keychains/System.keychain",
		filepath.Join(os.Getenv("HOME"), "/Library/Keychains/login.keychain"),
		filepath.Join(os.Getenv("HOME"), "/Library/Keychains/login.keychain-db")).Output()
	if err != nil {
		t.Fatal(err)
	}
	allCerts := NewCertPool()
	allCerts.AppendCertsFromPEM(out)

	// Check that the two cert pools are the same.
	sysPool := make(map[string]*Certificate, len(sysRoots.certs))
	for _, c := range sysRoots.certs {
		sysPool[string(c.Raw)] = c
	}
	for _, c := range execRoots.certs {
		if _, ok := sysPool[string(c.Raw)]; ok {
			delete(sysPool, string(c.Raw))
		} else {
			// verify-cert lets in certificates that are not trusted roots, but
			// are signed by trusted roots. This is not great, but unavoidable
			// until we parse real policies without cgo, so confirm that's the
			// case and skip them.
			if _, err := c.Verify(VerifyOptions{
				Roots:         sysRoots,
				Intermediates: allCerts,
				KeyUsages:     []ExtKeyUsage{ExtKeyUsageAny},
				CurrentTime:   c.NotBefore, // verify-cert does not check expiration
			}); err != nil {
				t.Errorf("certificate only present in non-cgo pool: %v (verify error: %v)", c.Subject, err)
			} else {
				t.Logf("signed certificate only present in non-cgo pool (acceptable): %v", c.Subject)
			}
		}
	}
	for _, c := range sysPool {
		// The nocgo codepath uses verify-cert with the ssl policy, which also
		// happens to check EKUs, so some certificates will appear only in the
		// cgo pool. We can't easily make them consistent because the EKU check
		// is only applied to the certificates passed to verify-cert.
		var ekuOk bool
		for _, eku := range c.ExtKeyUsage {
			if eku == ExtKeyUsageServerAuth || eku == ExtKeyUsageNetscapeServerGatedCrypto ||
				eku == ExtKeyUsageMicrosoftServerGatedCrypto || eku == ExtKeyUsageAny {
				ekuOk = true
			}
		}
		if len(c.ExtKeyUsage) == 0 && len(c.UnknownExtKeyUsage) == 0 {
			ekuOk = true
		}
		if !ekuOk {
			t.Logf("off-EKU certificate only present in cgo pool (acceptable): %v", c.Subject)
			continue
		}

		// Same for expired certificates. We don't chain to them anyway.
		now := time.Now()
		if now.Before(c.NotBefore) || now.After(c.NotAfter) {
			t.Logf("expired certificate only present in cgo pool (acceptable): %v", c.Subject)
			continue
		}

		// On 10.11 there are five unexplained roots that only show up from the
		// C API. They have in common the fact that they are old, 1024-bit
		// certificates. It's arguably better to ignore them anyway.
		if key, ok := c.PublicKey.(*rsa.PublicKey); ok && key.N.BitLen() == 1024 {
			t.Logf("1024-bit certificate only present in cgo pool (acceptable): %v", c.Subject)
			continue
		}

		t.Errorf("certificate only present in cgo pool: %v", c.Subject)
	}

	if t.Failed() && debugDarwinRoots {
		cmd := exec.Command("security", "dump-trust-settings")
		cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
		cmd.Run()
		cmd = exec.Command("security", "dump-trust-settings", "-d")
		cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
		cmd.Run()
	}
}
