// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"context"
	"fmt"
	"go/scanner"
	"go/token"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

// information needed for completion
type completer struct {
	p      *Parsed
	pos    protocol.Position
	offset int // offset of the start of the Token
	ctx    protocol.CompletionContext
	syms   map[string]symbol
}

func Completion(ctx context.Context, snapshot source.Snapshot, fh source.VersionedFileHandle, pos protocol.Position, context protocol.CompletionContext) (*protocol.CompletionList, error) {
	if skipTemplates(snapshot) {
		return nil, nil
	}
	all := New(snapshot.Templates())
	var start int // the beginning of the Token (completed or not)
	syms := make(map[string]symbol)
	var p *Parsed
	for fn, fc := range all.files {
		// collect symbols from all template files
		filterSyms(syms, fc.symbols)
		if fn.Filename() != fh.URI().Filename() {
			continue
		}
		if start = inTemplate(fc, pos); start == -1 {
			return nil, nil
		}
		p = fc
	}
	if p == nil {
		// this cannot happen unless the search missed a template file
		return nil, fmt.Errorf("%s not found", fh.FileIdentity().URI.Filename())
	}
	c := completer{
		p:      p,
		pos:    pos,
		offset: start + len(Left),
		ctx:    context,
		syms:   syms,
	}
	return c.complete()
}

func filterSyms(syms map[string]symbol, ns []symbol) {
	for _, xsym := range ns {
		switch xsym.kind {
		case protocol.Method, protocol.Package, protocol.Boolean, protocol.Namespace,
			protocol.Function:
			syms[xsym.name] = xsym // we don't care which symbol we get
		case protocol.Variable:
			if xsym.name != "dot" {
				syms[xsym.name] = xsym
			}
		case protocol.Constant:
			if xsym.name == "nil" {
				syms[xsym.name] = xsym
			}
		}
	}
}

// return the starting position of the enclosing token, or -1 if none
func inTemplate(fc *Parsed, pos protocol.Position) int {
	// 1. pos might be in a Token, return tk.Start
	// 2. pos might be after an elided but before a Token, return elided
	// 3. return -1 for false
	offset := fc.FromPosition(pos)
	// this could be a binary search, as the tokens are ordered
	for _, tk := range fc.tokens {
		if tk.Start <= offset && offset < tk.End {
			return tk.Start
		}
	}
	for _, x := range fc.elided {
		if x > offset {
			// fc.elided is sorted
			break
		}
		// If the interval [x,offset] does not contain Left or Right
		// then provide completions. (do we need the test for Right?)
		if !bytes.Contains(fc.buf[x:offset], []byte(Left)) && !bytes.Contains(fc.buf[x:offset], []byte(Right)) {
			return x
		}
	}
	return -1
}

var (
	keywords = []string{"if", "with", "else", "block", "range", "template", "end}}", "end"}
	globals  = []string{"and", "call", "html", "index", "slice", "js", "len", "not", "or",
		"urlquery", "printf", "println", "print", "eq", "ne", "le", "lt", "ge", "gt"}
)

// find the completions. start is the offset of either the Token enclosing pos, or where
// the incomplete token starts.
// The error return is always nil.
func (c *completer) complete() (*protocol.CompletionList, error) {
	ans := &protocol.CompletionList{IsIncomplete: true, Items: []protocol.CompletionItem{}}
	start := c.p.FromPosition(c.pos)
	sofar := c.p.buf[c.offset:start]
	if len(sofar) == 0 || sofar[len(sofar)-1] == ' ' || sofar[len(sofar)-1] == '\t' {
		return ans, nil
	}
	// sofar could be parsed by either c.analyzer() or scan(). The latter is precise
	// and slower, but fast enough
	words := scan(sofar)
	// 1. if pattern starts $, show variables
	// 2. if pattern starts ., show methods (and . by itself?)
	// 3. if len(words) == 1, show firstWords (but if it were a |, show functions and globals)
	// 4. ...? (parenthetical expressions, arguments, ...) (packages, namespaces, nil?)
	if len(words) == 0 {
		return nil, nil // if this happens, why were we called?
	}
	pattern := string(words[len(words)-1])
	if pattern[0] == '$' {
		// should we also return a raw "$"?
		for _, s := range c.syms {
			if s.kind == protocol.Variable && weakMatch(s.name, pattern) > 0 {
				ans.Items = append(ans.Items, protocol.CompletionItem{
					Label:  s.name,
					Kind:   protocol.VariableCompletion,
					Detail: "Variable",
				})
			}
		}
		return ans, nil
	}
	if pattern[0] == '.' {
		for _, s := range c.syms {
			if s.kind == protocol.Method && weakMatch("."+s.name, pattern) > 0 {
				ans.Items = append(ans.Items, protocol.CompletionItem{
					Label:  s.name,
					Kind:   protocol.MethodCompletion,
					Detail: "Method/member",
				})
			}
		}
		return ans, nil
	}
	// could we get completion attempts in strings or numbers, and if so, do we care?
	// globals
	for _, kw := range globals {
		if weakMatch(kw, string(pattern)) != 0 {
			ans.Items = append(ans.Items, protocol.CompletionItem{
				Label:  kw,
				Kind:   protocol.KeywordCompletion,
				Detail: "Function",
			})
		}
	}
	// and functions
	for _, s := range c.syms {
		if s.kind == protocol.Function && weakMatch(s.name, pattern) != 0 {
			ans.Items = append(ans.Items, protocol.CompletionItem{
				Label:  s.name,
				Kind:   protocol.FunctionCompletion,
				Detail: "Function",
			})
		}
	}
	// keywords if we're at the beginning
	if len(words) <= 1 || len(words[len(words)-2]) == 1 && words[len(words)-2][0] == '|' {
		for _, kw := range keywords {
			if weakMatch(kw, string(pattern)) != 0 {
				ans.Items = append(ans.Items, protocol.CompletionItem{
					Label:  kw,
					Kind:   protocol.KeywordCompletion,
					Detail: "keyword",
				})
			}
		}
	}
	return ans, nil
}

// someday think about comments, strings, backslashes, etc
// this would repeat some of the template parsing, but because the user is typing
// there may be no parse tree here.
// (go/scanner will report 2 tokens for $a, as $ is not a legal go identifier character)
// (go/scanner is about 2.7 times more expensive)
func (c *completer) analyze(buf []byte) [][]byte {
	// we want to split on whitespace and before dots
	var working []byte
	var ans [][]byte
	for _, ch := range buf {
		if ch == '.' && len(working) > 0 {
			ans = append(ans, working)
			working = []byte{'.'}
			continue
		}
		if ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' {
			if len(working) > 0 {
				ans = append(ans, working)
				working = []byte{}
				continue
			}
		}
		working = append(working, ch)
	}
	if len(working) > 0 {
		ans = append(ans, working)
	}
	ch := buf[len(buf)-1]
	if ch == ' ' || ch == '\t' {
		// avoid completing on whitespace
		ans = append(ans, []byte{ch})
	}
	return ans
}

// version of c.analyze that uses go/scanner.
func scan(buf []byte) []string {
	fset := token.NewFileSet()
	fp := fset.AddFile("", -1, len(buf))
	var sc scanner.Scanner
	sc.Init(fp, buf, func(pos token.Position, msg string) {}, scanner.ScanComments)
	ans := make([]string, 0, 10) // preallocating gives a measurable savings
	for {
		_, tok, lit := sc.Scan() // tok is an int
		if tok == token.EOF {
			break // done
		} else if tok == token.SEMICOLON && lit == "\n" {
			continue // don't care, but probably can't happen
		} else if tok == token.PERIOD {
			ans = append(ans, ".") // lit is empty
		} else if tok == token.IDENT && len(ans) > 0 && ans[len(ans)-1] == "." {
			ans[len(ans)-1] = "." + lit
		} else if tok == token.IDENT && len(ans) > 0 && ans[len(ans)-1] == "$" {
			ans[len(ans)-1] = "$" + lit
		} else {
			ans = append(ans, lit)
		}
	}
	return ans
}

// pattern is what the user has typed
func weakMatch(choice, pattern string) float64 {
	lower := strings.ToLower(choice)
	// for now, use only lower-case everywhere
	pattern = strings.ToLower(pattern)
	// The first char has to match
	if pattern[0] != lower[0] {
		return 0
	}
	// If they start with ., then the second char has to match
	from := 1
	if pattern[0] == '.' {
		if len(pattern) < 2 {
			return 1 // pattern just a ., so it matches
		}
		if pattern[1] != lower[1] {
			return 0
		}
		from = 2
	}
	// check that all the characters of pattern occur as a subsequence of choice
	for i, j := from, from; j < len(pattern); j++ {
		if pattern[j] == lower[i] {
			i++
			if i >= len(lower) {
				return 0
			}
		}
	}
	return 1
}

// for debug printing
func strContext(c protocol.CompletionContext) string {
	switch c.TriggerKind {
	case protocol.Invoked:
		return "invoked"
	case protocol.TriggerCharacter:
		return fmt.Sprintf("triggered(%s)", c.TriggerCharacter)
	case protocol.TriggerForIncompleteCompletions:
		// gopls doesn't seem to handle these explicitly anywhere
		return "incomplete"
	}
	return fmt.Sprintf("?%v", c)
}
