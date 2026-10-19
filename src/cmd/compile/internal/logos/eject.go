// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"os"
	"path/filepath"
	"strings"

	"cmd/compile/internal/syntax"
)

// Eject writes one ejected Go file per logos source into dir by applying
// edits to the original bytes. Files without a src entry are skipped.
// Every eject output comes from the same lowered nodes as the native
// build, so comments outside replacement spans survive verbatim.
func Eject(dir string, files []*syntax.File, src map[*syntax.File][]byte, edits map[*syntax.File][]Edit) error {
	if err := os.MkdirAll(dir, 0o777); err != nil {
		return err
	}
	for _, file := range files {
		data, ok := src[file]
		if !ok {
			continue
		}
		base := file.Pos().FileBase()
		out, err := ApplyEdits(data, base, edits[file])
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, ejectedName(base.Filename())), out, 0o666); err != nil {
			return err
		}
	}
	return nil
}

// ejectedName maps a logos file name to its ejected Go file name.
func ejectedName(name string) string {
	base := filepath.Base(name)
	return strings.TrimSuffix(base, filepath.Ext(base)) + ".go"
}
