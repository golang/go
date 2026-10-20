// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// EditMapSuffix is appended to the name of an ejected Go file to name its
// edit map: a.lgo ejects to a.go and a.go.edits.
const EditMapSuffix = ".edits"

// EditMapVersion is the version field of an EditMap. A reader rejects any
// other value.
const EditMapVersion = 1

// EditMap is the JSON document beside an ejected file. It lists the edits
// that turned the original source into the ejected text, as byte spans of
// the original, so a reader can map any offset of one text to the other
// (the language server proxy does this for positions). Source is the base
// name of the original file, Size and Out are the byte sizes of the
// original and ejected texts; a reader checks them against the texts it
// holds before it trusts the spans.
type EditMap struct {
	Version int        `json:"version"`
	Source  string     `json:"source"`
	Size    int        `json:"size"`
	Out     int        `json:"out"`
	Edits   []EditSpan `json:"edits"`
}

// writeEditMap writes the edit map of one ejected file to path. It uses
// encoding/json, not encoding/json/v2: the bootstrap toolchain builds this
// package, and v2 is an experiment there.
func writeEditMap(path, source string, size, out int, spans []EditSpan) error {
	data, err := json.Marshal(EditMap{
		Version: EditMapVersion,
		Source:  filepath.Base(source),
		Size:    size,
		Out:     out,
		Edits:   spans,
	})
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o666)
}
