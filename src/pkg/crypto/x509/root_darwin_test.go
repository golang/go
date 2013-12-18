package x509

import "testing"

func TestSystemRoots(t *testing.T) {
	sysRoots := systemRootsPool()         // actual system roots
	execRoots, err := execSecurityRoots() // non-cgo roots

	if err != nil {
		t.Fatalf("failed to read system roots: %v", err)
	}

	for _, tt := range []*CertPool{sysRoots, execRoots} {
		if tt == nil {
			t.Fatal("no system roots")
		}
		// On Mavericks, there are 212 bundled certs; require only
		// 150 here, since this is just a sanity check, and the
		// exact number will vary over time.
		if want, have := 150, len(tt.certs); have < want {
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
		t.Errorf("insufficent overlap between cgo and non-cgo roots; want at least %d, have %d", want, have)
	}
}
