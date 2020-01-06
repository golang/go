// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for vendoring semantics.

package main_test

import (
	"internal/testenv"
	"os"
	"testing"
)

func TestLegacyModGet(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExecPath(t, "git")

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("d1"))
	tg.run("get", "vcs-test.golang.org/git/modlegacy1-old.git/p1")
	tg.run("list", "-f", "{{.Deps}}", "vcs-test.golang.org/git/modlegacy1-old.git/p1")
	tg.grepStdout("new.git/p2", "old/p1 should depend on new/p2")
	tg.grepStdoutNot("new.git/v2/p2", "old/p1 should NOT depend on new/v2/p2")
	tg.run("build", "vcs-test.golang.org/git/modlegacy1-old.git/p1", "vcs-test.golang.org/git/modlegacy1-new.git/p1")

	tg.setenv("GOPATH", tg.path("d2"))

	tg.must(os.RemoveAll(tg.path("d2")))
	tg.run("get", "github.com/rsc/vgotest5")
	tg.run("get", "github.com/rsc/vgotest4")
	tg.run("get", "github.com/myitcv/vgo_example_compat")

	if testing.Short() {
		return
	}

	tg.must(os.RemoveAll(tg.path("d2")))
	tg.run("get", "github.com/rsc/vgotest4")
	tg.run("get", "github.com/rsc/vgotest5")
	tg.run("get", "github.com/myitcv/vgo_example_compat")

	tg.must(os.RemoveAll(tg.path("d2")))
	tg.run("get", "github.com/rsc/vgotest4", "github.com/rsc/vgotest5")
	tg.run("get", "github.com/myitcv/vgo_example_compat")

	tg.must(os.RemoveAll(tg.path("d2")))
	tg.run("get", "github.com/rsc/vgotest5", "github.com/rsc/vgotest4")
	tg.run("get", "github.com/myitcv/vgo_example_compat")

	tg.must(os.RemoveAll(tg.path("d2")))
	tg.run("get", "github.com/myitcv/vgo_example_compat")
	tg.run("get", "github.com/rsc/vgotest4", "github.com/rsc/vgotest5")

	pkgs := []string{"github.com/myitcv/vgo_example_compat", "github.com/rsc/vgotest4", "github.com/rsc/vgotest5"}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 3; k++ {
				if i == j || i == k || k == j {
					continue
				}
				tg.must(os.RemoveAll(tg.path("d2")))
				tg.run("get", pkgs[i], pkgs[j], pkgs[k])
			}
		}
	}
}
