// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package protocol

import (
	"encoding/json"
	"fmt"
)

// DocumentChanges is a union of a file edit and directory rename operations
// for package renaming feature. At most one field of this struct is non-nil.
type DocumentChanges struct {
	TextDocumentEdit *TextDocumentEdit
	RenameFile       *RenameFile
}

func (d *DocumentChanges) UnmarshalJSON(data []byte) error {
	var m map[string]interface{}

	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}

	if _, ok := m["textDocument"]; ok {
		d.TextDocumentEdit = new(TextDocumentEdit)
		return json.Unmarshal(data, d.TextDocumentEdit)
	}

	d.RenameFile = new(RenameFile)
	return json.Unmarshal(data, d.RenameFile)
}

func (d *DocumentChanges) MarshalJSON() ([]byte, error) {
	if d.TextDocumentEdit != nil {
		return json.Marshal(d.TextDocumentEdit)
	} else if d.RenameFile != nil {
		return json.Marshal(d.RenameFile)
	}
	return nil, fmt.Errorf("Empty DocumentChanges union value")
}
