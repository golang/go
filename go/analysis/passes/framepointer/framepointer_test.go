// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package framepointer_test

import (
	"go/build"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/framepointer"
)

func Test(t *testing.T) {
	if build.Default.GOOS != "linux" && build.Default.GOOS != "darwin" {
		// The test has an os-generic assembly file, testdata/a/asm_amd64.s.
		// It should produce errors on linux or darwin, but not on other archs.
		// Unfortunately, there's no way to say that in the "want" comments
		// in that file. So we skip testing on other GOOSes. The framepointer
		// analyzer should not report any errors on those GOOSes, so it's not
		// really a hard test on those platforms.
		t.Skipf("test for GOOS=%s is not implemented", build.Default.GOOS)
	}
	analysistest.Run(t, analysistest.TestData(), framepointer.Analyzer, "a")
}
