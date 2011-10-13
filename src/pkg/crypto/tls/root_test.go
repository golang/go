// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"testing"
)

var tlsServers = []string{
	"google.com:443",
	"github.com:443",
	"twitter.com:443",
}

func TestOSCertBundles(t *testing.T) {
	defaultRoots()

	if testing.Short() {
		t.Logf("skipping certificate tests in short mode")
		return
	}

	for _, addr := range tlsServers {
		conn, err := Dial("tcp", addr, nil)
		if err != nil {
			t.Errorf("unable to verify %v: %v", addr, err)
			continue
		}
		err = conn.Close()
		if err != nil {
			t.Error(err)
		}
	}
}
