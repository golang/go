// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"testing"

	"cmd/compile/internal/base"
	"internal/buildcfg"
)

func TestLogosPointeeActiveInLgoWithoutFlag(t *testing.T) {
	oldBase, oldExp := base.Logos, buildcfg.Experiment.LogosEscape
	t.Cleanup(func() {
		base.Logos = oldBase
		buildcfg.Experiment.LogosEscape = oldExp
	})
	base.Logos = true
	buildcfg.Experiment.LogosEscape = false
	if !logosPointeeActive() {
		t.Fatal("logosPointeeActive = false for a .lgo package without the experiment flag, want true")
	}
}

func TestLogosPointeeActiveUnderFlag(t *testing.T) {
	oldBase, oldExp := base.Logos, buildcfg.Experiment.LogosEscape
	t.Cleanup(func() {
		base.Logos = oldBase
		buildcfg.Experiment.LogosEscape = oldExp
	})
	base.Logos = false
	buildcfg.Experiment.LogosEscape = true
	if !logosPointeeActive() {
		t.Fatal("logosPointeeActive = false for a Go package with GOEXPERIMENT=logosescape, want true")
	}
}

func TestLogosPointeeInactiveByDefault(t *testing.T) {
	oldBase, oldExp := base.Logos, buildcfg.Experiment.LogosEscape
	t.Cleanup(func() {
		base.Logos = oldBase
		buildcfg.Experiment.LogosEscape = oldExp
	})
	base.Logos = false
	buildcfg.Experiment.LogosEscape = false
	if logosPointeeActive() {
		t.Fatal("logosPointeeActive = true for a Go package without the experiment flag, want false")
	}
}
