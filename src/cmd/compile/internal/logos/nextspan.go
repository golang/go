// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/syntax"
)

type spanScanner struct {
	src      []byte
	end      int
	firstEnd int
	lineEnd  int
	trailing string
	i        int
}

func scanComments(src []byte, start, firstEnd, end int) (string, bool) {
	s := &spanScanner{src: src, end: end, firstEnd: firstEnd, i: start}
	s.markLineEnd(start)
	for s.i < s.end {
		if s.step() {
			return "", true
		}
	}
	return s.trailing, false
}

func (s *spanScanner) markLineEnd(start int) {
	s.lineEnd = len(s.src)
	for i := start; i < len(s.src); i++ {
		if s.src[i] == '\n' {
			s.lineEnd = i
			break
		}
	}
}

func (s *spanScanner) step() bool {
	c := s.src[s.i]
	if c == '"' || c == '`' {
		s.i = skipQuoted(s.src, s.i)
		return false
	}
	if c == '\'' {
		s.i = skipChar(s.src, s.i)
		return false
	}
	if c == '/' && s.i+1 < s.end {
		return s.slash()
	}
	s.i++
	return false
}

func (s *spanScanner) slash() bool {
	if s.src[s.i+1] == '/' {
		return s.lineComment()
	}
	if s.src[s.i+1] == '*' {
		return true
	}
	s.i++
	return false
}

func (s *spanScanner) lineComment() bool {
	ls := s.i
	for s.i < s.end && s.src[s.i] != '\n' {
		s.i++
	}
	if ls < s.lineEnd && s.firstEnd <= s.lineEnd && isGapComment(s.src[s.firstEnd:ls]) {
		s.trailing = string(s.src[s.firstEnd:s.lineEnd])
		return false
	}
	return true
}

func isGapComment(gap []byte) bool {
	for _, c := range gap {
		if c != ' ' && c != '\t' {
			return false
		}
	}
	return true
}

func skipQuoted(src []byte, i int) int {
	q := src[i]
	i++
	for i < len(src) {
		if q != '`' && src[i] == '\\' {
			i += 2
			continue
		}
		if src[i] == q {
			return i + 1
		}
		if q != '`' && src[i] == '\n' {
			return i
		}
		i++
	}
	return i
}

func skipChar(src []byte, i int) int {
	i++
	if i < len(src) && src[i] == '\\' {
		i++
	}
	for i < len(src) && src[i] != '\'' && src[i] != '\n' {
		i++
	}
	if i < len(src) && src[i] == '\'' {
		i++
	}
	return i
}

func (s *nextScan) endOf(e syntax.Expr) (int, bool) {
	switch n := e.(type) {
	case *syntax.Name:
		return s.after(n.Pos(), len(n.Value))
	case *syntax.BasicLit:
		return s.after(n.Pos(), len(n.Value))
	case *syntax.SelectorExpr:
		return s.endOf(n.Sel)
	case *syntax.ParenExpr:
		return s.endOfParen(n)
	case *syntax.CallExpr:
		return s.endOfCall(n)
	case *syntax.IndexExpr:
		return s.endOfIndexed(n.X)
	case *syntax.SliceExpr:
		return s.endOfIndexed(n.X)
	case *syntax.Operation:
		return s.endOfOp(n)
	default:
		return s.endOfComposite(e)
	}
}

func (s *nextScan) endOfParen(n *syntax.ParenExpr) (int, bool) {
	off, err := offsetOf(s.starts, len(s.src), n.Pos())
	if err != nil {
		return 0, false
	}
	return matchClose(s.src, off, '(')
}

func (s *nextScan) endOfCall(n *syntax.CallExpr) (int, bool) {
	fun, ok := s.endOf(n.Fun)
	if !ok {
		return 0, false
	}
	return s.closeFrom(fun, '(')
}

func (s *nextScan) endOfIndexed(x syntax.Expr) (int, bool) {
	start, ok := s.endOf(x)
	if !ok {
		return 0, false
	}
	return s.closeFrom(start, '[')
}

func (s *nextScan) endOfOp(n *syntax.Operation) (int, bool) {
	if n.Y != nil {
		return s.endOf(n.Y)
	}
	return s.endOf(n.X)
}

func (s *nextScan) endOfComposite(e syntax.Expr) (int, bool) {
	switch n := e.(type) {
	case *syntax.ListExpr:
		return s.endOfList(n)
	case *syntax.CompositeLit:
		return s.after(n.Rbrace, 1)
	case *syntax.FuncLit:
		return s.endOfFunc(n)
	case *syntax.KeyValueExpr:
		return s.endOf(n.Value)
	case *syntax.AssertExpr:
		return s.endOf(n.Type)
	case *syntax.SliceType:
		return s.endOf(n.Elem)
	case *syntax.ArrayType:
		return s.endOf(n.Elem)
	}
	return 0, false
}

func (s *nextScan) endOfList(n *syntax.ListExpr) (int, bool) {
	if len(n.ElemList) == 0 {
		return 0, false
	}
	return s.endOf(n.ElemList[len(n.ElemList)-1])
}

func (s *nextScan) endOfFunc(n *syntax.FuncLit) (int, bool) {
	if n.Body == nil {
		return 0, false
	}
	return s.after(n.Body.Rbrace, 1)
}

func (s *nextScan) callText(call *syntax.CallExpr) (string, bool) {
	start, ok := s.startOf(call)
	if !ok {
		return "", false
	}
	end, ok := s.endOf(call)
	if !ok || end < start {
		return "", false
	}
	return string(s.src[start:end]), true
}

func (s *nextScan) startOf(e syntax.Expr) (int, bool) {
	switch n := e.(type) {
	case *syntax.CallExpr:
		return s.startOf(n.Fun)
	case *syntax.IndexExpr:
		return s.startOf(n.X)
	case *syntax.SelectorExpr:
		return s.startOf(n.X)
	case *syntax.SliceExpr:
		return s.startOf(n.X)
	case *syntax.AssertExpr:
		return s.startOf(n.X)
	case *syntax.CompositeLit:
		if n.Type != nil {
			return s.startOf(n.Type)
		}
	case *syntax.Operation:
		if n.Y != nil {
			return s.startOf(n.X)
		}
	}
	start, err := offsetOf(s.starts, len(s.src), e.Pos())
	if err != nil {
		return 0, false
	}
	return start, true
}

func (s *nextScan) after(pos syntax.Pos, n int) (int, bool) {
	start, err := offsetOf(s.starts, len(s.src), pos)
	if err != nil {
		return 0, false
	}
	if start+n > len(s.src) {
		return 0, false
	}
	return start + n, true
}

func (s *nextScan) closeFrom(off int, open byte) (int, bool) {
	i := off
	for i < len(s.src) && (s.src[i] == ' ' || s.src[i] == '\t' || s.src[i] == '\n' || s.src[i] == '\r') {
		i++
	}
	if i >= len(s.src) || s.src[i] != open {
		return 0, false
	}
	return matchClose(s.src, i, open)
}

type closeScanner struct {
	src   []byte
	want  byte
	close byte
	depth int
	i     int
}

func matchClose(src []byte, open int, want byte) (int, bool) {
	close, ok := closeFor(want)
	if !ok {
		return 0, false
	}
	s := &closeScanner{src: src, want: want, close: close, i: open}
	for s.i < len(s.src) {
		if end, ok := s.step(); ok {
			return end, true
		}
	}
	return 0, false
}

func closeFor(want byte) (byte, bool) {
	switch want {
	case '(':
		return ')', true
	case '[':
		return ']', true
	case '{':
		return '}', true
	}
	return 0, false
}

func (s *closeScanner) step() (int, bool) {
	c := s.src[s.i]
	if c == '"' || c == '`' {
		s.i = skipQuoted(s.src, s.i)
		return 0, false
	}
	if c == '\'' {
		s.i = skipChar(s.src, s.i)
		return 0, false
	}
	if s.skipComment() {
		return 0, false
	}
	return s.bracket(c)
}

func (s *closeScanner) skipComment() bool {
	if s.src[s.i] != '/' || s.i+1 >= len(s.src) {
		return false
	}
	if s.src[s.i+1] == '/' {
		s.skipLineComment()
		return true
	}
	if s.src[s.i+1] != '*' {
		return false
	}
	s.skipBlockComment()
	return true
}

func (s *closeScanner) skipLineComment() {
	for s.i < len(s.src) && s.src[s.i] != '\n' {
		s.i++
	}
}

func (s *closeScanner) skipBlockComment() {
	s.i += 2
	for s.i+1 < len(s.src) && (s.src[s.i] != '*' || s.src[s.i+1] != '/') {
		s.i++
	}
	s.i += 2
}

func (s *closeScanner) bracket(c byte) (int, bool) {
	switch c {
	case s.want:
		s.depth++
	case s.close:
		s.depth--
		if s.depth == 0 {
			s.i++
			return s.i, true
		}
	}
	s.i++
	return 0, false
}
