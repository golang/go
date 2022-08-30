// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"context"
	"fmt"
	"regexp"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
)

func Highlight(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle, loc protocol.Position) ([]protocol.DocumentHighlight, error) {
	buf, err := fh.Read()
	if err != nil {
		return nil, err
	}
	p := parseBuffer(buf)
	pos := p.FromPosition(loc)
	var ans []protocol.DocumentHighlight
	if p.ParseErr == nil {
		for _, s := range p.symbols {
			if s.start <= pos && pos < s.start+s.length {
				return markSymbols(p, s)
			}
		}
	}
	// these tokens exist whether or not there was a parse error
	// (symbols require a successful parse)
	for _, tok := range p.tokens {
		if tok.Start <= pos && pos < tok.End {
			wordAt := findWordAt(p, pos)
			if len(wordAt) > 0 {
				return markWordInToken(p, wordAt)
			}
		}
	}
	// find the 'word' at pos, etc: someday
	// until then we get the default action, which doesn't respect word boundaries
	return ans, nil
}

func markSymbols(p *Parsed, sym symbol) ([]protocol.DocumentHighlight, error) {
	var ans []protocol.DocumentHighlight
	for _, s := range p.symbols {
		if s.name == sym.name {
			kind := protocol.Read
			if s.vardef {
				kind = protocol.Write
			}
			ans = append(ans, protocol.DocumentHighlight{
				Range: p.Range(s.start, s.length),
				Kind:  kind,
			})
		}
	}
	return ans, nil
}

// A token is {{...}}, and this marks words in the token that equal the give word
func markWordInToken(p *Parsed, wordAt string) ([]protocol.DocumentHighlight, error) {
	var ans []protocol.DocumentHighlight
	pat, err := regexp.Compile(fmt.Sprintf(`\b%s\b`, wordAt))
	if err != nil {
		return nil, fmt.Errorf("%q: unmatchable word (%v)", wordAt, err)
	}
	for _, tok := range p.tokens {
		got := pat.FindAllIndex(p.buf[tok.Start:tok.End], -1)
		for i := 0; i < len(got); i++ {
			ans = append(ans, protocol.DocumentHighlight{
				Range: p.Range(got[i][0], got[i][1]-got[i][0]),
				Kind:  protocol.Text,
			})
		}
	}
	return ans, nil
}

var wordRe = regexp.MustCompile(`[$]?\w+$`)
var moreRe = regexp.MustCompile(`^[$]?\w+`)

// findWordAt finds the word the cursor is in (meaning in or just before)
func findWordAt(p *Parsed, pos int) string {
	if pos >= len(p.buf) {
		return "" // can't happen, as we are called with pos < tok.End
	}
	after := moreRe.Find(p.buf[pos:])
	if len(after) == 0 {
		return "" // end of the word
	}
	got := wordRe.Find(p.buf[:pos+len(after)])
	return string(got)
}
