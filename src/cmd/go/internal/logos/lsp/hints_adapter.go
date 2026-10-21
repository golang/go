// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"encoding/json"

	"cmd/go/internal/logos/hints"
)

// codeActionsAdapter is the CodeActionsFunc runLSP serves: it decodes the
// compiler diagnostic lines and returns the quick fixes hints.CodeActions
// builds for the request range as a JSON array, [] when there are none. A
// line that is not a diagnostic of the schema version hints reads is
// skipped, so one bad line never costs the fixes of the others. Only an
// unreadable range is an error.
func codeActionsAdapter(diags [][]byte, uri string, src []byte, rng json.RawMessage) (json.RawMessage, error) {
	var r hints.Range
	if err := json.Unmarshal(rng, &r); err != nil {
		return nil, err
	}
	ds := make([]hints.Diagnostic, 0, len(diags))
	for _, line := range diags {
		d, err := hints.DecodeDiagnostic(line)
		if err != nil {
			continue
		}
		ds = append(ds, d)
	}
	acts := hints.CodeActions(ds, uri, src, r)
	if acts == nil {
		acts = []hints.CodeAction{}
	}
	return json.Marshal(acts)
}
