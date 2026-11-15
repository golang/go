// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/build"
	"internal/testenv"
	"path/filepath"
	"slices"
	"testing"

	"cmd/internal/sys"
)

func TestViewAsCopiesOnTargetsWithoutUnalignedLoads(t *testing.T) {
	dir := filepath.Join(testenv.GOROOT(t), "src", "logos", "rt")
	for _, arch := range sys.Archs {
		ctx := build.Default
		ctx.GOOS, ctx.GOARCH = "linux", arch.Name
		if arch.Name == "wasm" {
			ctx.GOOS = "js"
		}
		pkg, err := ctx.ImportDir(dir, 0)
		if err != nil {
			t.Fatalf("%s: import logos/rt: %v", arch.Name, err)
		}
		direct := slices.Contains(pkg.GoFiles, "viewas_direct.go")
		copied := slices.Contains(pkg.GoFiles, "viewas_copy.go")
		if direct == copied {
			t.Errorf("%s: logos/rt must select exactly one of viewas_direct.go and viewas_copy.go, got direct=%t copy=%t", arch.Name, direct, copied)
		}
		if direct && arch.Alignment != 1 {
			t.Errorf("%s: ViewAs addresses the caller's bytes directly, but the backend requires %d-byte alignment for smaller loads", arch.Name, arch.Alignment)
		}
	}
}
