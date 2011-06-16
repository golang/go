// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"exec"
	"path/filepath"
	"testing"
)

var buildPkgs = []string{
	"go/build/pkgtest",
	"go/build/cmdtest",
	"go/build/cgotest",
}

const cmdtestOutput = "3"

func TestBuild(t *testing.T) {
	for _, pkg := range buildPkgs {
		tree := Path[0] // Goroot
		dir := filepath.Join(tree.SrcDir(), pkg)

		info, err := ScanDir(dir, true)
		if err != nil {
			t.Error("ScanDir:", err)
			continue
		}

		s, err := Build(tree, pkg, info)
		if err != nil {
			t.Error("Build:", err)
			continue
		}

		if err := s.Run(); err != nil {
			t.Error("Run:", err)
			continue
		}

		if pkg == "go/build/cmdtest" {
			bin := s.Output[0]
			b, err := exec.Command(bin).CombinedOutput()
			if err != nil {
				t.Errorf("exec: %s: %v", bin, err)
				continue
			}
			if string(b) != cmdtestOutput {
				t.Errorf("cmdtest output: %s want: %s", b, cmdtestOutput)
			}
		}

		defer func(s *Script) {
			if err := s.Nuke(); err != nil {
				t.Errorf("nuking: %v", err)
			}
		}(s)
	}
}
