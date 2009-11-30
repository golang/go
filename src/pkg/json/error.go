// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import "fmt"

// ParseError aggregates information about a JSON parse error.  It is
// compatible with the os.Error interface.
type ParseError struct {
	Index	int;	// A byte index in JSON string where the error occurred
	Token	string;	// An offending token
}

// Produce a string representation of this ParseError.
func (pe *ParseError) String() string {
	return fmt.Sprintf("Unexpected JSON token at position %d: %q.", pe.Index, pe.Token)
}
