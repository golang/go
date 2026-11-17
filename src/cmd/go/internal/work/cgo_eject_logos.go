// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"fmt"
	"internal/logosfile"
	"path/filepath"
	"slices"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/fsys"
)

// logosEjectCgo checks and lowers original C sources in a separate
// frontend process. The ordinary compile follows with cmd/cgo's rewritten
// sources, retaining native pointer checks and the authoritative archive.
// Its ejected non-C files and support metadata replace the source pass's
// copies; original C sources survive without generated cgo identifiers.
func logosEjectCgo(sh *Shell, a *Action, args []any, gofiles []string) ([]byte, error) {
	if !logosEjectNoCache(a) {
		return nil, nil
	}
	p := a.Package
	originals := make(map[string]string)
	for _, name := range p.CgoFiles {
		if logosfile.Is(name) {
			originals[filepath.Base(name)] = name
		}
	}
	if len(originals) == 0 {
		return nil, nil
	}
	var inputs, rewrites []string
	for _, name := range gofiles {
		if stem, ok := strings.CutSuffix(filepath.Base(name), ".cgo1.go"); ok {
			if source, ok := originals[stem]; ok {
				name = source
				delete(originals, stem)
			}
		}
		logical := mkAbs(p.Dir, name)
		actual := fsys.Actual(logical)
		inputs = append(inputs, actual)
		if actual != logical {
			rewrites = append(rewrites, actual+"=>"+logical)
		}
	}
	for _, name := range p.CgoFiles {
		if _, missing := originals[filepath.Base(name)]; missing {
			return nil, fmt.Errorf("logos eject: missing cgo compiler input for %s", name)
		}
	}
	// This process writes source, not object paths. Recover the logical
	// overlay names before source classification, without the native
	// directory rewrites that may match a backing file first.
	args = append(slices.Clone(args), "-logos.cgoeject", "-trimpath", strings.Join(rewrites, ";"), inputs)
	return sh.runOut(base.Cwd(), cfgChangedEnv, args...)
}
