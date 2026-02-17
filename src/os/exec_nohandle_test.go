// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux && !windows

package os_test

import (
	"internal/testenv"
	. "os"
	"testing"
	"time"
)

func TestProcessWithHandleUnsupported(t *testing.T) {
	const envVar = "OSTEST_PROCESS_WITH_HANDLE"
	if Getenv(envVar) != "" {
		time.Sleep(1 * time.Minute)
		return
	}

	cmd := testenv.CommandContext(t, t.Context(), testenv.Executable(t), "-test.run=^"+t.Name()+"$")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, envVar+"=1")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		cmd.Process.Kill()
		cmd.Wait()
	}()

	err := cmd.Process.WithHandle(func(handle uintptr) {
		t.Errorf("WithHandle: callback called unexpectedly with handle=%v", handle)
	})
	if err != ErrNoHandle {
		t.Fatalf("WithHandle: got error %v, want %v", err, ErrNoHandle)
	}
}
