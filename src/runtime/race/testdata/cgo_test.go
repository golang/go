// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"internal/testenv"
	"os"
	"os/exec"
	"testing"
)

func TestNoRaceCgoSync(t *testing.T) {
	cmd := exec.Command(testenv.GoToolPath(t), "run", "-race", "cgo_test_main.go")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("program exited with error: %v\n", err)
	}
}
