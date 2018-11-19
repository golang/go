// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sanitizers_test

import (
	"runtime"
	"strings"
	"testing"
)

func TestTSAN(t *testing.T) {
	if runtime.GOARCH == "arm64" {
		t.Skip("skipping test; see https://golang.org/issue/25682")
	}
	t.Parallel()
	requireOvercommit(t)
	config := configure("thread")
	config.skipIfCSanitizerBroken(t)

	mustRun(t, config.goCmd("build", "std"))

	cases := []struct {
		src          string
		needsRuntime bool
	}{
		{src: "tsan.go"},
		{src: "tsan2.go"},
		{src: "tsan3.go"},
		{src: "tsan4.go"},
		{src: "tsan5.go", needsRuntime: true},
		{src: "tsan6.go", needsRuntime: true},
		{src: "tsan7.go", needsRuntime: true},
		{src: "tsan8.go"},
		{src: "tsan9.go"},
		{src: "tsan10.go", needsRuntime: true},
		{src: "tsan11.go", needsRuntime: true},
		{src: "tsan12.go", needsRuntime: true},
	}
	for _, tc := range cases {
		tc := tc
		name := strings.TrimSuffix(tc.src, ".go")
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			dir := newTempDir(t)
			defer dir.RemoveAll(t)

			outPath := dir.Join(name)
			mustRun(t, config.goCmd("build", "-o", outPath, srcPath(tc.src)))

			cmd := hangProneCmd(outPath)
			if tc.needsRuntime {
				config.skipIfRuntimeIncompatible(t)
			}
			mustRun(t, cmd)
		})
	}
}
