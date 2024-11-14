// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"testing"
)

func TestFallbackPanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Multiple calls to SetFallbackRoots should panic")
		}
	}()
	SetFallbackRoots(nil)
	SetFallbackRoots(nil)
}

func TestFallback(t *testing.T) {
	// call systemRootsPool so that the sync.Once is triggered, and we can
	// manipulate systemRoots without worrying about our working being overwritten
	systemRootsPool()
	if systemRoots != nil {
		originalSystemRoots := *systemRoots
		defer func() { systemRoots = &originalSystemRoots }()
	}

	tests := []struct {
		name            string
		systemRoots     *CertPool
		systemPool      bool
		poolContent     []*Certificate
		forceFallback   bool
		returnsFallback bool
	}{
		{
			name:            "nil systemRoots",
			returnsFallback: true,
		},
		{
			name:            "empty systemRoots",
			systemRoots:     NewCertPool(),
			returnsFallback: true,
		},
		{
			name:        "empty systemRoots system pool",
			systemRoots: NewCertPool(),
			systemPool:  true,
		},
		{
			name:        "filled systemRoots system pool",
			systemRoots: NewCertPool(),
			poolContent: []*Certificate{{}},
			systemPool:  true,
		},
		{
			name:        "filled systemRoots",
			systemRoots: NewCertPool(),
			poolContent: []*Certificate{{}},
		},
		{
			name:            "filled systemRoots, force fallback",
			systemRoots:     NewCertPool(),
			poolContent:     []*Certificate{{}},
			forceFallback:   true,
			returnsFallback: true,
		},
		{
			name:            "filled systemRoot system pool, force fallback",
			systemRoots:     NewCertPool(),
			poolContent:     []*Certificate{{}},
			systemPool:      true,
			forceFallback:   true,
			returnsFallback: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func { t ->
			fallbacksSet = false
			systemRoots = tc.systemRoots
			if systemRoots != nil {
				systemRoots.systemPool = tc.systemPool
			}
			for _, c := range tc.poolContent {
				systemRoots.AddCert(c)
			}
			if tc.forceFallback {
				t.Setenv("GODEBUG", "x509usefallbackroots=1")
			} else {
				t.Setenv("GODEBUG", "x509usefallbackroots=0")
			}

			fallbackPool := NewCertPool()
			SetFallbackRoots(fallbackPool)

			systemPoolIsFallback := systemRoots == fallbackPool

			if tc.returnsFallback && !systemPoolIsFallback {
				t.Error("systemRoots was not set to fallback pool")
			} else if !tc.returnsFallback && systemPoolIsFallback {
				t.Error("systemRoots was set to fallback pool when it shouldn't have been")
			}
		})
	}
}
