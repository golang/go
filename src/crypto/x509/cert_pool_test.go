// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import "testing"

func TestCertPoolEqual(t *testing.T) {
	a, b := NewCertPool(), NewCertPool()
	if !a.Equal(b) {
		t.Error("two empty pools not equal")
	}

	tc := &Certificate{Raw: []byte{1, 2, 3}, RawSubject: []byte{2}}
	a.AddCert(tc)
	if a.Equal(b) {
		t.Error("empty pool equals non-empty pool")
	}

	b.AddCert(tc)
	if !a.Equal(b) {
		t.Error("two non-empty pools not equal")
	}

	otherTC := &Certificate{Raw: []byte{9, 8, 7}, RawSubject: []byte{8}}
	a.AddCert(otherTC)
	if a.Equal(b) {
		t.Error("non-equal pools equal")
	}

	systemA, err := SystemCertPool()
	if err != nil {
		t.Fatalf("unable to load system cert pool: %s", err)
	}
	systemB, err := SystemCertPool()
	if err != nil {
		t.Fatalf("unable to load system cert pool: %s", err)
	}
	if !systemA.Equal(systemB) {
		t.Error("two empty system pools not equal")
	}

	systemA.AddCert(tc)
	if systemA.Equal(systemB) {
		t.Error("empty system pool equals non-empty system pool")
	}

	systemB.AddCert(tc)
	if !systemA.Equal(systemB) {
		t.Error("two non-empty system pools not equal")
	}

	systemA.AddCert(otherTC)
	if systemA.Equal(systemB) {
		t.Error("non-equal system pools equal")
	}
}
