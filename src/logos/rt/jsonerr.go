// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"strconv"
	"strings"
)

// MissingFieldError is the error a generated UnmarshalJSON returns when a
// required non-nil field is absent from the object or holds JSON null
// (SPEC Sec 10.2, T-Sec10). Type is the struct type name and Field is the
// Go field name.
type MissingFieldError struct {
	Type  string
	Field string
}

// Error names the struct type and the field.
func (e MissingFieldError) Error() string {
	return "logos: json: missing required non-nil field " + e.Type + "." + e.Field
}

// UnionTrialError is the error a generated UnmarshalJSON returns when
// trial unmarshaling of a closed union fails because zero members matched
// or more than one member matched (ambiguity) (SPEC Sec 10.2, T-Sec10, T-203.1).
type UnionTrialError struct {
	Union   string
	Members []string
	Matched int
	Total   int
}

// Error names the union type, member names, and trial outcome counts.
func (e UnionTrialError) Error() string {
	members := strings.Join(e.Members, ", ")
	if e.Matched > 1 {
		return "logos: json: ambiguous unmarshal into union " + e.Union + ": " + strconv.Itoa(e.Matched) + " of " + strconv.Itoa(e.Total) + " members matched (" + members + ")"
	}
	return "logos: json: cannot unmarshal into union " + e.Union + ": 0 of " + strconv.Itoa(e.Total) + " members matched (" + members + ")"
}
