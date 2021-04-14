// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"os"
	"os/exec"
	"testing"
	"time"
)

func TestSystemRoots(t *testing.T) {
	t0 := time.Now()
	sysRoots, err := loadSystemRoots() // actual system roots
	sysRootsDuration := time.Since(t0)

	if err != nil {
		t.Fatalf("failed to read system roots: %v", err)
	}

	t.Logf("loadSystemRoots: %v", sysRootsDuration)

	// There are 174 system roots on Catalina, and 163 on iOS right now, require
	// at least 100 to make sure this is not completely broken.
	if want, have := 100, sysRoots.len(); have < want {
		t.Errorf("want at least %d system roots, have %d", want, have)
	}

	if t.Failed() {
		cmd := exec.Command("security", "dump-trust-settings")
		cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
		cmd.Run()
		cmd = exec.Command("security", "dump-trust-settings", "-d")
		cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
		cmd.Run()
	}
}
