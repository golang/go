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

const allowMarker = "//logos:allow"

// allowDirective matches //logos:allow(<rule>), optionally followed by a
// space and free text. Both parts allow digits after the first letter so the
// helper's canonical nogo::<lowercase-analyzer> IDs (for example,
// nogo::sa1019, nogo::printf) parse with the same scope logic as the native
// group::name IDs, which remain lowercase without digits. Uppercase stays
// rejected: the driver lowercases helper categories (SA1019 to sa1019)
// before forming IDs, so only the lowercase canonical form reaches here.
// An own-line comment that attempts the directive but fails to match is
// not silently dropped: fileDirectives records it as malformed and Run
// reports correctness::allow_malformed, so a typo cannot silently leave a
// rule enabled while the author believes it allowed.
var allowDirective = regexp.MustCompile(`^//logos:allow\(([a-z][a-z0-9_]*::[a-z][a-z0-9_]*)\)(?:\s.*)?$`)

// AllowForbiddenRule is the ID of the diagnostic for a //logos:allow
// directive that names a forbid rule in a loGo/s package.
const AllowForbiddenRule = "correctness::allow_forbidden_rule"

// AllowMalformedRule is the ID of the diagnostic for a //logos:allow
// comment that fails to parse. It names no rule, silences nothing, and
// always reports at forbid (in Go packages too) so a typo cannot silently
// leave a rule enabled while the author believes it allowed.
const AllowMalformedRule = "correctness::allow_malformed"

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
	// text is the raw comment of a malformed directive; empty otherwise.
	text string
}

// directives holds the directives of one package, keyed by file base,
// plus the malformed //logos:allow comments of every file in source order.
// Malformed entries silence nothing: silences consults byFile only.
type directives struct {
	byFile    map[*syntax.PosBase][]directive
	malformed []directive
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
		if !ok || !bytes.Contains(src, []byte(allowMarker)) {
			continue
		}
		fileBase := f.Pos().FileBase()
		dirs, bad := fileDirectives(f, src, fileBase, mode)
		ds.byFile[fileBase] = dirs
		ds.malformed = append(ds.malformed, bad...)
	}
	return ds
}

// isAllowComment reports whether text attempts a //logos:allow directive:
// the word followed by "(", white space, or nothing. Longer words such as
// //logos:allowlist stay out of the malformed set.
func isAllowComment(text string) bool {
	if _, ok := syntax.DirectiveArg(text, allowMarker); ok {
		return true
	}
	return strings.HasPrefix(text, allowMarker+"(")
}

// fileDirectives returns the directives of one file in source order,
// scanning its comments with the given parser mode. The second result
// holds the malformed //logos:allow comments: own-line attempts that fail
// to parse, such as an uppercase group, a missing "::", or a missing
// parenthesized ID. Well-formed but unknown IDs (for example the external
// nogo:: helpers) are valid directives, not malformed: they parse with
// the same scope logic and simply silence no native rule.
func fileDirectives(f *syntax.File, src []byte, fileBase *syntax.PosBase, mode syntax.Mode) (out, bad []directive) {
	lines := strings.Split(string(src), "\n")
	starts := nodeStarts(f)
	syntax.CommentsDoMode(bytes.NewReader(src), mode, func(line, col uint, text string) {
		if !isAllowComment(text) || !ownLine(lines, line, col) {
			return
		}
		trimmed := strings.TrimRight(text, " \t\r")
		pos := syntax.MakePos(fileBase, line, col)
		end := syntax.MakePos(fileBase, line, col+uint(len(trimmed)))
		m := allowDirective.FindStringSubmatch(text)
		if m == nil {
			bad = append(bad, directive{pos: pos, end: end, text: trimmed})
			return
		}
		d := directive{rule: m[1], pos: pos, end: end}
		target := nextCodeLine(lines, line)
		if lastLine, ok := starts[target]; ok {
			d.scope, d.from, d.to = true, line+1, lastLine
		}
		out = append(out, d)
	})
	return out, bad
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
// that names a rule reporting at forbid in a package of the given kind,
// plus one allow_malformed result for each malformed directive. Both are
// directive misuses that Run reports without consulting silencing, so
// neither can silently disable a rule. Resolve (failed-check packages)
// reports neither: it applies only severity and silencing to the findings
// of earlier phases.
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
	return append(out, ds.malformedResults()...)
}

// malformedResults returns one allow_malformed result for each malformed
// //logos:allow comment. The severity stays forbid in every package kind:
// malformed syntax is not a language-neutral lint, so the Go cap that
// lowers forbid findings to deny does not apply. A malformed comment names
// no rule and therefore cannot silence one; the result is always reported.
func (ds *directives) malformedResults() []Result {
	var out []Result
	for _, d := range ds.malformed {
		out = append(out, Result{
			Rule:     AllowMalformedRule,
			Severity: Forbid,
			Pos:      d.pos,
			End:      d.end,
			Message:  d.text + ": malformed directive, want //logos:allow(<group>::<name>)",
		})
	}
	return out
}
