// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import "testing"

func TestCertPoolEqual(t *testing.T) {
	tc := &Certificate{Raw: []byte{1, 2, 3}, RawSubject: []byte{2}}
	otherTC := &Certificate{Raw: []byte{9, 8, 7}, RawSubject: []byte{8}}

	emptyPool := NewCertPool()
	nonSystemPopulated := NewCertPool()
	nonSystemPopulated.AddCert(tc)
	nonSystemPopulatedAlt := NewCertPool()
	nonSystemPopulatedAlt.AddCert(otherTC)
	emptySystem, err := SystemCertPool()
	if err != nil {
		t.Fatal(err)
	}
	populatedSystem, err := SystemCertPool()
	if err != nil {
		t.Fatal(err)
	}
	populatedSystem.AddCert(tc)
	populatedSystemAlt, err := SystemCertPool()
	if err != nil {
		t.Fatal(err)
	}
	populatedSystemAlt.AddCert(otherTC)
	tests := []struct {
		name  string
		a     *CertPool
		b     *CertPool
		equal bool
	}{
		{
			name:  "two empty pools",
			a:     emptyPool,
			b:     emptyPool,
			equal: true,
		},
		{
			name:  "one empty pool, one populated pool",
			a:     emptyPool,
			b:     nonSystemPopulated,
			equal: false,
		},
		{
			name:  "two populated pools",
			a:     nonSystemPopulated,
			b:     nonSystemPopulated,
			equal: true,
		},
		{
			name:  "two populated pools, different content",
			a:     nonSystemPopulated,
			b:     nonSystemPopulatedAlt,
			equal: false,
		},
		{
			name:  "two empty system pools",
			a:     emptySystem,
			b:     emptySystem,
			equal: true,
		},
		{
			name:  "one empty system pool, one populated system pool",
			a:     emptySystem,
			b:     populatedSystem,
			equal: false,
		},
		{
			name:  "two populated system pools",
			a:     populatedSystem,
			b:     populatedSystem,
			equal: true,
		},
		{
			name:  "two populated pools, different content",
			a:     populatedSystem,
			b:     populatedSystemAlt,
			equal: false,
		},
		{
			name:  "two nil pools",
			a:     nil,
			b:     nil,
			equal: true,
		},
		{
			name:  "one nil pool, one empty pool",
			a:     nil,
			b:     emptyPool,
			equal: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func { t ->
			equal := tc.a.Equal(tc.b)
			if equal != tc.equal {
				t.Errorf("Unexpected Equal result: got %t, want %t", equal, tc.equal)
			}
		})
	}
}
