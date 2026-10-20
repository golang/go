// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"bytes"
	"regexp"
	"strings"

	"cmd/compile/internal/syntax"
)

// allowDirective matches //logos:allow(<rule>), optionally followed by a
// space and free text.
var allowDirective = regexp.MustCompile(`^//logos:allow\(([a-z][a-z_]*::[a-z][a-z_]*)\)(?:\s.*)?$`)

// AllowForbiddenRule is the ID of the diagnostic for a //logos:allow
// directive that names a forbid rule in a loGo/s package.
const AllowForbiddenRule = "correctness::allow_forbidden_rule"

// directive is one //logos:allow comment and the statement or declaration
// it precedes. The scope runs from the line after the directive through the
// last line of the statements and declarations starting on the first line
// after the comment block, so it covers a comment between the directive and
// that node; it is empty when no node starts there. Positions inside a node
// compare by line because the syntax package places a node at its operator
// or name.
type directive struct {
	rule     string
	pos      syntax.Pos
	end      syntax.Pos
	scope    bool
	from, to uint
}

// directives holds the directives of one package, keyed by file base.
type directives struct {
	byFile map[*syntax.PosBase][]directive
}

// collectDirectives scans the comments of every file that has source.
func collectDirectives(in *Input) *directives {
	ds := &directives{byFile: make(map[*syntax.PosBase][]directive)}
	mode := syntax.Mode(0)
	if in.Kind == KindLgo {
		mode = syntax.LogosMode
	}
	for _, f := range in.Files {
		src, ok := in.Sources[f]
		if !ok || !bytes.Contains(src, []byte("//logos:allow(")) {
			continue
		}
		fileBase := f.Pos().FileBase()
		ds.byFile[fileBase] = fileDirectives(f, src, fileBase, mode)
	}
	return ds
}

// fileDirectives returns the directives of one file in source order,
// scanning its comments with the given parser mode.
func fileDirectives(f *syntax.File, src []byte, fileBase *syntax.PosBase, mode syntax.Mode) []directive {
	lines := strings.Split(string(src), "\n")
	starts := nodeStarts(f)
	var out []directive
	syntax.CommentsDoMode(bytes.NewReader(src), mode, func(line, col uint, text string) {
		m := allowDirective.FindStringSubmatch(text)
		if m == nil || !ownLine(lines, line, col) {
			return
		}
		d := directive{
			rule: m[1],
			pos:  syntax.MakePos(fileBase, line, col),
			end:  syntax.MakePos(fileBase, line, col+uint(len(strings.TrimRight(text, " \t\r")))),
		}
		target := nextCodeLine(lines, line)
		if lastLine, ok := starts[target]; ok {
			d.scope, d.from, d.to = true, line+1, lastLine
		}
		out = append(out, d)
	})
	return out
}

// ownLine reports whether the comment at (line, col) is the first token on
// its line, so it can precede a statement.
func ownLine(lines []string, line, col uint) bool {
	if line == 0 || int(line) > len(lines) || col == 0 {
		return false
	}
	return strings.TrimSpace(lines[line-1][:col-1]) == ""
}

// nextCodeLine returns the first line after line that is not a line
// comment. A blank line ends the comment block: it returns 0.
func nextCodeLine(lines []string, line uint) uint {
	for l := line + 1; int(l) <= len(lines); l++ {
		text := strings.TrimSpace(lines[l-1])
		switch {
		case text == "":
			return 0
		case strings.HasPrefix(text, "//"):
			continue
		}
		return l
	}
	return 0
}

// nodeStarts maps each line on which a statement or declaration starts to
// the last line of the longest such node.
func nodeStarts(f *syntax.File) map[uint]uint {
	starts := make(map[uint]uint)
	syntax.Inspect(f, func(n syntax.Node) bool {
		if n == nil {
			return false
		}
		if isStmtOrDecl(n) {
			first := syntax.StartPos(n).Line()
			starts[first] = max(starts[first], syntax.EndPos(n).Line())
		}
		return true
	})
	return starts
}

// isStmtOrDecl reports whether n is a statement or a declaration.
func isStmtOrDecl(n syntax.Node) bool {
	switch n.(type) {
	case syntax.Decl, syntax.Stmt:
		return true
	}
	return false
}

// silences reports whether a directive for rule precedes the statement or
// declaration containing pos.
func (ds *directives) silences(rule string, pos syntax.Pos) bool {
	for _, d := range ds.byFile[pos.FileBase()] {
		if d.rule != rule || !d.scope {
			continue
		}
		if pos.Line() >= d.from && pos.Line() <= d.to {
			return true
		}
	}
	return false
}

// forbidden returns one allow_forbidden_rule result for each directive
// that names a rule reporting at forbid in a package of the given kind.
func (ds *directives) forbidden(policy *Policy, kind Kind) []Result {
	var out []Result
	for _, list := range ds.byFile {
		for _, d := range list {
			if policy.Level(d.rule, kind) != Forbid {
				continue
			}
			out = append(out, Result{
				Rule:     AllowForbiddenRule,
				Severity: Forbid,
				Pos:      d.pos,
				End:      d.end,
				Message:  "//logos:allow(" + d.rule + "): rule " + d.rule + " is forbid and cannot be allowed",
			})
		}
	}
	return out
}
