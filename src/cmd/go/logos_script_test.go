// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"cmd/internal/script"
	"errors"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

var testLogos string

func setupTestLogos() {
	testLogos = filepath.Join(testBin, "logos"+exeSuffix)
	if err := os.Link(testGo, testLogos); err != nil {
		logosSrc, err := os.Open(testGo)
		if err != nil {
			log.Fatal(err)
		}
		logosDst, err := os.OpenFile(testLogos, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o777)
		if err != nil {
			logosSrc.Close()
			log.Fatal(err)
		}
		_, copyErr := io.Copy(logosDst, logosSrc)
		closeErr := logosDst.Close()
		srcErr := logosSrc.Close()
		if copyErr != nil {
			log.Fatal(copyErr)
		}
		if closeErr != nil {
			log.Fatal(closeErr)
		}
		if srcErr != nil {
			log.Fatal(srcErr)
		}
	}
}

func scriptGoNamed(path string, cancel func(*exec.Cmd) error, waitDelay time.Duration) script.Cmd {
	cmd := script.Program(path, cancel, waitDelay)
	// Inject code to update scriptGoInvoked before invoking the Go command.
	return script.Command(*cmd.Usage(), func(state *script.State, s ...string) (script.WaitFunc, error) {
		t, ok := tbFromContext(state.Context())
		if !ok {
			return nil, errors.New("script Context unexpectedly missing testing.TB key")
		}
		_, dup := scriptGoInvoked.LoadOrStore(t, true)
		if !dup {
			t.Cleanup(func() { scriptGoInvoked.Delete(t) })
		}
		return cmd.Run(state, s...)
	})
}
