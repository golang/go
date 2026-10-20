// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"bytes"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// todoPlaceholder is slop::todo_placeholder (SPEC Sec 1.2): unfinished work
// left in a non-test file. Two forms count. A comment whose text, after the
// comment marker and any white space, starts with the word TODO or FIXME; a
// word ends at a character that cannot be part of an identifier, so TODOS
// and FIXMEs are prose. A call of the predeclared panic whose argument is a
// constant string holding "unimplemented" or "not implemented" in any letter
// case, or "TODO" as written. Comments come from the scanner by
// syntax.CommentsDoMode, the form of syntax.CommentsDo that reads a loGo/s
// file with its own token rules, so a // inside an interpolated string is
// never a comment. A file whose name ends in _test is skipped.
var todoPlaceholder = Rule{
	ID:    "slop::todo_placeholder",
	Group: Slop,
	Go:    true,
	Check: checkTodoPlaceholder,
}

// checkTodoPlaceholder is the Check of todoPlaceholder.
func checkTodoPlaceholder(pass *Pass) {
	for _, f := range pass.Files {
		if IsTestFile(f) {
			continue
		}
		todoComments(pass, f)
		todoPanics(pass, f)
	}
}

// todoComments reports each comment of f that starts with a marker. A file
// without source bytes, or whose bytes hold neither word, is not scanned: a
// marker is case-sensitive, so no comment of such a file can start with one.
func todoComments(pass *Pass, f *syntax.File) {
	src := pass.Source(f)
	if !bytes.Contains(src, []byte("TODO")) && !bytes.Contains(src, []byte("FIXME")) {
		return
	}
	mode := syntax.Mode(0)
	if pass.Kind == KindLgo {
		mode = syntax.LogosMode
	}
	base := f.Pos().FileBase()
	syntax.CommentsDoMode(bytes.NewReader(src), mode, func(line, col uint, text string) {
		marker := todoMarker(text)
		if marker == "" {
			return
		}
		pass.Report(Diagnostic{
			Pos:        syntax.MakePos(base, line, col),
			End:        todoCommentEnd(base, line, col, text),
			Message:    "comment starts with " + marker + ": finish the work or remove the marker",
			Provenance: Provenance{OriginExpr: text, OriginLine: int(line)},
		})
	})
}

// todoCommentEnd returns the position just after the comment text that starts
// at line and col.
func todoCommentEnd(base *syntax.PosBase, line, col uint, text string) syntax.Pos {
	newlines := strings.Count(text, "\n")
	if newlines == 0 {
		return syntax.MakePos(base, line, col+uint(len(text)))
	}
	return syntax.MakePos(base, line+uint(newlines), uint(len(text)-strings.LastIndexByte(text, '\n')))
}

// todoMarker returns TODO or FIXME when the scanner's text is a comment
// that starts with that word, and "" otherwise. Text that does not start
// with a slash is a scanner error message, not a comment.
func todoMarker(text string) string {
	if !strings.HasPrefix(text, "//") && !strings.HasPrefix(text, "/*") {
		return ""
	}
	body := strings.TrimLeftFunc(text[2:], unicode.IsSpace)
	for _, marker := range []string{"TODO", "FIXME"} {
		if rest, ok := strings.CutPrefix(body, marker); ok && !todoStartsWord(rest) {
			return marker
		}
	}
	return ""
}

// todoStartsWord reports whether s starts with a letter, digit, or underscore.
func todoStartsWord(s string) bool {
	r, _ := utf8.DecodeRuneInString(s)
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)
}

// todoPanics reports each panic stub of f.
func todoPanics(pass *Pass, f *syntax.File) {
	syntax.Inspect(f, func(n syntax.Node) bool {
		call, ok := n.(*syntax.CallExpr)
		if !ok {
			return true
		}
		if msg, stub := todoPanicStub(pass, call); stub {
			pass.Report(Diagnostic{
				Pos:        syntax.StartPos(call),
				End:        pass.CallEnd(f, call),
				Message:    "panic stub " + strconv.Quote(msg) + ": implement it or return an error",
				Provenance: Provenance{OriginExpr: syntax.String(call), OriginLine: int(call.Pos().Line())},
			})
		}
		return true
	})
}

// todoPanicStub returns the message of a call of the predeclared panic whose
// only argument is a constant string that names unfinished work.
func todoPanicStub(pass *Pass, call *syntax.CallExpr) (string, bool) {
	name, ok := call.Fun.(*syntax.Name)
	if !ok || len(call.ArgList) != 1 {
		return "", false
	}
	if b, ok := pass.Info.Uses[name].(*types2.Builtin); !ok || b.Name() != "panic" {
		return "", false
	}
	msg, ok := stringConstOf(call.ArgList[0])
	return msg, ok && todoNamesUnfinished(msg)
}

// todoNamesUnfinished reports whether a panic message says the code is
// unimplemented, not implemented, or carries the marker TODO.
func todoNamesUnfinished(msg string) bool {
	lower := strings.ToLower(msg)
	return strings.Contains(lower, "unimplemented") ||
		strings.Contains(lower, "not implemented") ||
		strings.Contains(msg, "TODO")
}
