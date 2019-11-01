// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tests

import (
	"fmt"
	"go/token"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

// DiffLinks takes the links we got and checks if they are located within the source or a Note.
// If the link is within a Note, the link is removed.
// Returns an diff comment if there are differences and empty string if no diffs
func DiffLinks(mapper *protocol.ColumnMapper, wantLinks []Link, gotLinks []protocol.DocumentLink) string {
	var notePositions []token.Position
	links := make(map[span.Span]string, len(wantLinks))
	for _, link := range wantLinks {
		links[link.Src] = link.Target
		notePositions = append(notePositions, link.NotePosition)
	}
	for _, link := range gotLinks {
		spn, err := mapper.RangeSpan(link.Range)
		if err != nil {
			return fmt.Sprintf("%v", err)
		}
		linkInNote := false
		for _, notePosition := range notePositions {
			// Drop the links found inside expectation notes arguments as this links are not collected by expect package
			if notePosition.Line == spn.Start().Line() &&
				notePosition.Column <= spn.Start().Column() {
				delete(links, spn)
				linkInNote = true
			}
		}
		if linkInNote {
			continue
		}
		if target, ok := links[spn]; ok {
			delete(links, spn)
			if target != link.Target {
				return fmt.Sprintf("for %v want %v, got %v\n", spn, link.Target, target)
			}
		} else {
			return fmt.Sprintf("unexpected link %v:%v\n", spn, link.Target)
		}
	}
	for spn, target := range links {
		return fmt.Sprintf("missing link %v:%v\n", spn, target)
	}
	return ""
}
