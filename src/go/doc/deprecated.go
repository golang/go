// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import "strings"

// Deprecated returns the deprecation notice, if any, in a doc comment.
// A deprecation notice is the text of a paragraph that starts with "Deprecated: ",
// stripped of the "Deprecated: " prefix.
// If there is no deprecation notice, it returns "", false.
//
// Doc can be [Func.Doc], [Package.Doc], [Type.Doc], or [Value.Doc].
func Deprecated(doc string) (text string, ok bool) {
	const marker = "Deprecated: "
	for len(doc) > 0 {
		var para string
		if i := strings.Index(doc, "\n\n"); i >= 0 {
			para, doc = doc[:i], doc[i+2:]
		} else {
			para, doc = doc, ""
		}
		if rest, found := strings.CutPrefix(para, marker); found {
			return strings.TrimSuffix(rest, "\n"), true
		}
	}
	return "", false
}
