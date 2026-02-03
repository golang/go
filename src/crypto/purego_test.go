// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crypto_test

import (
	"go/build"
	"internal/testenv"
	"log"
	"os"
	"strings"
	"testing"
)

// TestPureGoTag checks that when built with the purego build tag, crypto
// packages don't require any assembly. This is used by alternative compilers
// such as TinyGo. See also the "crypto/...:purego" test in cmd/dist, which
// ensures the packages build correctly.
func TestPureGoTag(t *testing.T) {
	cmd := testenv.Command(t, testenv.GoToolPath(t), "list", "-e", "crypto/...", "math/big")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Environ(), "GOOS=linux", "GOFIPS140=off")
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		log.Fatalf("loading package list: %v\n%s", err, out)
	}
	pkgs := strings.Split(strings.TrimSpace(string(out)), "\n")

	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "dist", "list")
	cmd.Stderr = os.Stderr
	out, err = testenv.CleanCmdEnv(cmd).Output()
	if err != nil {
		log.Fatalf("loading architecture list: %v\n%s", err, out)
	}
	allGOARCH := make(map[string]bool)
	for _, pair := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		GOARCH := strings.Split(pair, "/")[1]
		allGOARCH[GOARCH] = true
	}

	for _, pkgName := range pkgs {
		if strings.Contains(pkgName, "/boring") {
			continue
		}

		for GOARCH := range allGOARCH {
			context := build.Context{
				GOOS:      "linux", // darwin has custom assembly
				GOARCH:    GOARCH,
				GOROOT:    testenv.GOROOT(t),
				Compiler:  build.Default.Compiler,
				BuildTags: []string{"purego", "math_big_pure_go"},
			}

			pkg, err := context.Import(pkgName, "", 0)
			if err != nil {
				t.Fatal(err)
			}
			if len(pkg.SFiles) == 0 {
				continue
			}
			t.Errorf("package %s has purego assembly files on %s: %v", pkgName, GOARCH, pkg.SFiles)
		}
	}
}
