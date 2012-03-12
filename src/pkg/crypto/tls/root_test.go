// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/x509"
	"runtime"
	"testing"
)

var tlsServers = []string{
	"google.com",
	"github.com",
	"twitter.com",
}

func TestOSCertBundles(t *testing.T) {
	if testing.Short() {
		t.Logf("skipping certificate tests in short mode")
		return
	}

	for _, addr := range tlsServers {
		conn, err := Dial("tcp", addr+":443", &Config{ServerName: addr})
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

func TestCertHostnameVerifyWindows(t *testing.T) {
	if runtime.GOOS != "windows" {
		return
	}

	if testing.Short() {
		t.Logf("skipping certificate tests in short mode")
		return
	}

	for _, addr := range tlsServers {
		cfg := &Config{ServerName: "example.com"}
		conn, err := Dial("tcp", addr+":443", cfg)
		if err == nil {
			conn.Close()
			t.Errorf("should fail to verify for example.com: %v", addr)
			continue
		}
		_, ok := err.(x509.HostnameError)
		if !ok {
			t.Errorf("error type mismatch, got: %v", err)
		}
	}
}
