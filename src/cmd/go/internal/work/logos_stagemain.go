// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"os"

	"cmd/go/internal/load"
)

func (b *Builder) StageTestmain(pmain *load.Package) error {
	testDir := b.CompileAction(ModeBuild, ModeBuild, pmain).Objdir
	if err := os.MkdirAll(testDir, 0777); err != nil {
		return err
	}
	pmain.Dir = testDir
	return os.WriteFile(testDir+"_testmain.go", *pmain.Internal.TestmainGo, 0666)
}
