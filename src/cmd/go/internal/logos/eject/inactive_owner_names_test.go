// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/internal/logosowned"
)

func TestInactiveOwnedRequestPackageName(t *testing.T) {
	for _, name := range []string{"windowsowner", "invented"} {
		t.Run(name, func(t *testing.T) {
			dir, out, staging := t.TempDir(), t.TempDir(), t.TempDir()
			const path = "example.com/mod/sub"
			host := ownedTestMainPackage(dir, path, "linuxowner")
			variant := ownedTestMainPackage(dir, path, "windowsowner")
			plan := ownedTestPlan(dir, host, variant)
			req := ownedTestRequest(t, path, name, "package "+name+"\nfunc Generated() {}\n", []logosowned.Declaration{})
			writeOwnedSidecar(t, staging, path, req)
			_, err := emitOwnedHelpers(t.Context(), plan, staging, out, map[string]string{})
			if name == "invented" {
				if err == nil || !strings.Contains(err.Error(), `owned request package "invented" does not match owner package "linuxowner"`) {
					t.Fatalf("unloaded package name accepted or lost diagnostic: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("loaded alternate package name rejected: %v", err)
			}
			files, err := filepath.Glob(filepath.Join(out, "sub", "logos_owned_*.go"))
			if err != nil || len(files) != 1 {
				t.Fatalf("expected one helper, got %v: %v", files, err)
			}
			data, err := os.ReadFile(files[0])
			if err != nil || !strings.Contains(string(data), "package windowsowner") {
				t.Fatalf("helper lost selected package name: %q, %v", data, err)
			}
		})
	}
}

func TestInactiveOwnedGroupsPackageNames(t *testing.T) {
	for _, sameName := range []bool{false, true} {
		name := "distinct"
		if sameName {
			name = "same"
		}
		t.Run(name, func(t *testing.T) {
			dir, out := t.TempDir(), t.TempDir()
			const path = "example.com/mod/sub"
			first := ownedTestMainPackage(dir, path, "linuxowner")
			second := ownedTestMainPackage(dir, path, "windowsowner")
			if sameName {
				second.Name = first.Name
			}
			plan := ownedTestPlan(dir, first, second)
			linux := ownedTestRequest(t, path, first.Name, "//go:build linux\n\npackage "+first.Name+"\ntype Shared int\n", []logosowned.Declaration{})
			windows := ownedTestRequest(t, path, second.Name, "//go:build windows\n\npackage "+second.Name+"\ntype Shared string\n", []logosowned.Declaration{})
			writers, err := applyOwnedConditionsAndCollisions(t.Context(), plan, []*ownedDest{
				{req: &linux, owner: first, dstRoot: out, dstDir: out},
				{req: &windows, owner: second, dstRoot: out, dstDir: out},
			})
			if sameName {
				if err == nil || !strings.Contains(err.Error(), `both declare "Shared"`) {
					t.Fatalf("same-package collision accepted or lost diagnostic: %v", err)
				}
				return
			}
			if err != nil || len(writers) != 2 {
				t.Fatalf("disjoint package clauses collided: writers=%d error=%v", len(writers), err)
			}
			for _, writer := range writers {
				wantTag := "linux"
				if writer.owner == second {
					wantTag = "windows"
				}
				if !strings.Contains(string(writer.fixed), "//go:build "+wantTag) || !strings.Contains(string(writer.fixed), "package "+writer.owner.Name) {
					t.Fatalf("helper lost package or platform condition: %s", writer.fixed)
				}
			}
		})
	}
}
