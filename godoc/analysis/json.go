// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analysis

// This file defines types used by client-side JavaScript.

type anchorJSON struct {
	Text string // HTML
	Href string // URL
}

// Indicates one of these forms of fact about a type T:
// T "is implemented by <ByKind> type <Other>"  (ByKind != "", e.g. "array")
// T "implements <Other>"                       (ByKind == "")
type implFactJSON struct {
	ByKind string `json:",omitempty"`
	Other  anchorJSON
}

// Implements facts are grouped by form, for ease of reading.
type implGroupJSON struct {
	Descr string
	Facts []implFactJSON
}

// JavaScript's onClickIdent() expects a TypeInfoJSON.
type TypeInfoJSON struct {
	Name        string // type name
	Size, Align int64
	Methods     []anchorJSON
	ImplGroups  []implGroupJSON
}

// JavaScript's cgAddChild requires a global array of PCGNodeJSON
// called CALLGRAPH, representing the intra-package call graph.
// The first element is special and represents "all external callers".
type PCGNodeJSON struct {
	Func    anchorJSON
	Callees []int // indices within CALLGRAPH of nodes called by this one
}
