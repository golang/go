// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"bytes"
	"encoding/json"
	"internal/testenv"
	"os"
	"os/exec"
	"testing"
)

// FetchModule fetches the module at the given version and returns the directory
// containing its source tree. It skips the test if fetching modules is not
// possible in this environment.
func FetchModule(t *testing.T, module, version string) string {
	testenv.MustHaveExternalNetwork(t)

	// If the default GOMODCACHE doesn't exist, use a temporary directory
	// instead. (For example, run.bash sets GOPATH=/nonexist-gopath.)
	out, err := testenv.CleanCmdEnv(testenv.Command(t, testenv.GoToolPath(t), "env", "GOMODCACHE")).Output()
	if err != nil {
		t.Errorf("%s env GOMODCACHE: %v\n%s", testenv.GoToolPath(t), err, out)
		if ee, ok := err.(*exec.ExitError); ok {
			t.Logf("%s", ee.Stderr)
		}
		t.FailNow()
	}
	modcacheOk := false
	if gomodcache := string(bytes.TrimSpace(out)); gomodcache != "" {
		if _, err := os.Stat(gomodcache); err == nil {
			modcacheOk = true
		}
	}
	if !modcacheOk {
		t.Setenv("GOMODCACHE", t.TempDir())
		// Allow t.TempDir() to clean up subdirectories.
		t.Setenv("GOFLAGS", os.Getenv("GOFLAGS")+" -modcacherw")
	}

	t.Logf("fetching %s@%s\n", module, version)

	output, err := testenv.Command(t, testenv.GoToolPath(t), "mod", "download", "-json", module+"@"+version).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to download %s@%s: %s\n%s\n", module, version, err, output)
	}
	var j struct {
		Dir string
	}
	if err := json.Unmarshal(output, &j); err != nil {
		t.Fatalf("failed to parse 'go mod download': %s\n%s\n", err, output)
	}

	return j.Dir
}
