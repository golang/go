// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmp"
	"go/constant"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// errChainReset is slop::err_chain_reset (SPEC Sec 1.2): a fmt.Errorf call
// that formats an operand of an error type with a verb other than %w, or
// that passes err.Error() or a string concatenated from it, so errors.Is and
// errors.As cannot see the cause through the new error. The verbs come from
// the constant format string, with the operand each one formats found the way
// package fmt finds it: a * width or precision and an explicit [n] index move
// the operand, and %% takes none. A call whose arguments are spread with ...
// is skipped, and a call whose format is not constant has no known operand
// per verb, so only an err.Error() argument can flag it. A finding offers one
// fix that replaces the verb of every offending operand with %w, when the
// format is a string literal and each offender has such a verb.
var errChainReset = Rule{
	ID:    "slop::err_chain_reset",
	Group: Slop,
	Go:    true,
	Check: checkErrChainReset,
}

// checkErrChainReset is the Check of errChainReset.
func checkErrChainReset(pass *Pass) {
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			call, ok := n.(*syntax.CallExpr)
			if ok && chainIsErrorf(pass, call) {
				c := &chainCall{pass: pass, file: f, call: call}
				if d, found := c.diagnose(); found {
					pass.Report(d)
				}
			}
			return true
		})
	}
}

// chainIsErrorf reports whether call calls fmt.Errorf.
func chainIsErrorf(pass *Pass, call *syntax.CallExpr) bool {
	fn := calleeFunc(pass, call)
	return fn != nil && fn.FullName() == "fmt.Errorf"
}

// chainCall is the analysis of one fmt.Errorf call.
type chainCall struct {
	pass *Pass
	file *syntax.File
	call *syntax.CallExpr
}

// chainFault is one operand of a fmt.Errorf call that cuts the error chain:
// verb is the verb that formats it, or nil when errCall, a call of an Error
// method, is the argument or one of its concatenated parts.
type chainFault struct {
	arg     int
	verb    *chainVerb
	errCall *syntax.CallExpr
}

// diagnose returns the finding for the call, if it cuts an error chain.
func (c *chainCall) diagnose() (Diagnostic, bool) {
	args := c.call.ArgList
	if len(args) == 0 || c.call.HasDots {
		return Diagnostic{}, false
	}
	format, hasFormat := stringConstOf(args[0])
	var verbs []chainVerb
	if hasFormat {
		verbs = chainFormatVerbs(format)
	}
	faults := c.faults(verbs)
	if len(faults) == 0 {
		return Diagnostic{}, false
	}
	d := Diagnostic{
		Pos:     syntax.StartPos(c.call),
		End:     c.pass.CallEnd(c.file, c.call),
		Message: c.message(faults[0]),
		Provenance: Provenance{
			OriginExpr:   syntax.String(c.call),
			OriginLine:   int(c.call.Pos().Line()),
			InferredType: c.faultType(faults[0]),
		},
	}
	if text, ok := c.fixedCall(format, verbs, faults); ok {
		d.Fixes = []Fix{{Description: "replace the verb with %w", Replacement: text}}
	}
	return d, true
}

// faults returns the operands that cut the chain, in argument order.
func (c *chainCall) faults(verbs []chainVerb) []chainFault {
	args := c.call.ArgList
	var out []chainFault
	for i := range verbs {
		v := &verbs[i]
		if v.verb != 'w' && v.arg+1 < len(args) && chainIsError(args[v.arg+1]) {
			out = append(out, chainFault{arg: v.arg + 1, verb: v})
		}
	}
	for i, arg := range args {
		if call := c.errorString(arg); call != nil {
			out = append(out, chainFault{arg: i, errCall: call})
		}
	}
	slices.SortStableFunc(out, func(a, b chainFault) int { return cmp.Compare(a.arg, b.arg) })
	return out
}

// errorString returns the call of an Error method that e is, or that e
// concatenates with + into the string it builds.
func (c *chainCall) errorString(e syntax.Expr) *syntax.CallExpr {
	e = syntax.Unparen(e)
	if call := callOf(e); call != nil {
		if chainErrorMethodCall(c.pass, call) {
			return call
		}
		return nil
	}
	if op, ok := e.(*syntax.Operation); ok && op.Op == syntax.Add && op.Y != nil {
		if call := c.errorString(op.X); call != nil {
			return call
		}
		return c.errorString(op.Y)
	}
	return nil
}

// chainErrorMethodCall reports whether call is x.Error() for an x of an error
// type, an Error method with no argument that returns a string.
func chainErrorMethodCall(pass *Pass, call *syntax.CallExpr) bool {
	sel, ok := call.Fun.(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "Error" || len(call.ArgList) != 0 {
		return false
	}
	fn, _ := pass.Info.Uses[sel.Sel].(*types2.Func)
	if fn == nil {
		return false
	}
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.Params().Len() != 0 || sig.Results().Len() != 1 {
		return false
	}
	return types2.Identical(sig.Results().At(0).Type(), types2.Typ[types2.String]) && chainIsError(sel.X)
}

// chainErrorInterface is the method set of the predeclared error type.
var chainErrorInterface = types2.Universe.Lookup("error").Type().Underlying().(*types2.Interface)

// chainIsError reports whether the static type of e implements error.
func chainIsError(e syntax.Expr) bool {
	t := TypeOf(e)
	return t != nil && types2.Implements(t, chainErrorInterface)
}

// message describes one fault.
func (c *chainCall) message(f chainFault) string {
	if f.verb != nil {
		return "fmt.Errorf formats error operand " + syntax.String(c.call.ArgList[f.arg]) +
			" with " + f.verb.text + ": use %w to keep the error chain"
	}
	return "fmt.Errorf passes " + syntax.String(f.errCall) + ": pass the error itself with %w to keep the error chain"
}

// faultType returns the type of the faulty operand.
func (c *chainCall) faultType(f chainFault) string {
	e := c.call.ArgList[f.arg]
	if f.verb == nil {
		e = f.errCall.Fun.(*syntax.SelectorExpr).X
	}
	if t := TypeOf(e); t != nil {
		return types2.TypeString(t, types2.RelativeTo(c.pass.Pkg))
	}
	return ""
}

// fixedCall returns the text of the call with every faulty verb replaced by
// %w, an err.Error() argument replaced by err along with its verb. It
// reports false when the format is not a string literal or a fault has no
// verb to replace, such as an Error call inside a concatenation or one
// formatted with a verb that is not %s or %v.
func (c *chainCall) fixedCall(format string, verbs []chainVerb, faults []chainFault) (string, bool) {
	lit, ok := c.call.ArgList[0].(*syntax.BasicLit)
	if !ok || lit.Kind != syntax.StringLit {
		return "", false
	}
	var targets []chainVerb
	replaced := make(map[int]syntax.Expr)
	for _, f := range faults {
		v, recv, ok := c.fixTarget(verbs, f)
		if !ok {
			return "", false
		}
		targets = append(targets, v)
		if recv != nil {
			replaced[f.arg] = recv
		}
	}
	return c.callText(chainRequote(lit, chainSpliceVerbs(format, targets)), replaced), true
}

// callText prints the call with newFormat as its format and the arguments
// named in replaced swapped for their replacement.
func (c *chainCall) callText(newFormat string, replaced map[int]syntax.Expr) string {
	parts := make([]string, len(c.call.ArgList))
	for i, arg := range c.call.ArgList {
		parts[i] = syntax.String(arg)
		if recv, ok := replaced[i]; ok {
			parts[i] = syntax.String(recv)
		}
	}
	parts[0] = newFormat
	return syntax.String(c.call.Fun) + "(" + strings.Join(parts, ", ") + ")"
}

// fixTarget returns the verb to replace for fault f and, for an err.Error()
// argument, the expression that replaces the argument.
func (c *chainCall) fixTarget(verbs []chainVerb, f chainFault) (chainVerb, syntax.Expr, bool) {
	if f.verb != nil {
		return *f.verb, nil, true
	}
	arg := syntax.Unparen(c.call.ArgList[f.arg])
	if callOf(arg) != f.errCall {
		return chainVerb{}, nil, false
	}
	for _, v := range verbs {
		if v.arg+1 == f.arg && (v.verb == 's' || v.verb == 'v') {
			return v, f.errCall.Fun.(*syntax.SelectorExpr).X, true
		}
	}
	return chainVerb{}, nil, false
}

// chainSpliceVerbs returns format with the verb of each target replaced by w.
func chainSpliceVerbs(format string, targets []chainVerb) string {
	slices.SortFunc(targets, func(a, b chainVerb) int { return cmp.Compare(b.off, a.off) })
	for _, v := range slices.Compact(targets) {
		format = format[:v.off] + "w" + format[v.end:]
	}
	return format
}

// chainRequote spells value as a string literal in the style of lit.
func chainRequote(lit *syntax.BasicLit, value string) string {
	if strings.HasPrefix(lit.Value, "`") {
		return "`" + value + "`"
	}
	return strconv.Quote(value)
}

// stringConstOf returns the value of e when the checker found a constant
// string, such as a literal or a named constant.
func stringConstOf(e syntax.Expr) (string, bool) {
	v := e.GetTypeInfo().Value
	if v == nil || v.Kind() != constant.String {
		return "", false
	}
	return constant.StringVal(v), true
}

// chainVerb is one verb of a format string: text is the directive from its %
// through the verb, arg the zero-based operand the verb formats, and off and
// end the byte span of the verb inside the format.
type chainVerb struct {
	text     string
	verb     rune
	arg      int
	off, end int
}

// chainFormatVerbs returns the verbs of format that format an operand, in order.
func chainFormatVerbs(format string) []chainVerb {
	s := chainVerbScan{format: format}
	var out []chainVerb
	for {
		v, ok := s.next()
		if !ok {
			return out
		}
		if v.verb != '%' {
			out = append(out, v)
		}
	}
}

// chainVerbScan walks a format string the way package fmt does: flags, an
// optional [n] operand index, a width and a precision that may each be *,
// then the verb.
type chainVerbScan struct {
	format string
	pos    int
	arg    int
}

// next returns the next directive. It reports false at the end of the
// format or at a % with no verb.
func (s *chainVerbScan) next() (chainVerb, bool) {
	i := strings.IndexByte(s.format[s.pos:], '%')
	if i < 0 {
		return chainVerb{}, false
	}
	from := s.pos + i
	s.pos = from + 1
	s.skipFlags()
	indexed := s.indexedSize()
	if s.pos < len(s.format) && s.format[s.pos] == '.' {
		s.pos++
		indexed = s.indexedSize()
	}
	if !indexed {
		s.argIndex()
	}
	if s.pos >= len(s.format) {
		return chainVerb{}, false
	}
	r, size := utf8.DecodeRuneInString(s.format[s.pos:])
	v := chainVerb{verb: r, arg: s.arg, off: s.pos, end: s.pos + size}
	v.text = s.format[from:v.end]
	s.pos = v.end
	if r != '%' {
		s.arg++
	}
	return v, true
}

// argIndex reads an explicit [n] operand index and reports whether one was
// there. A malformed index is skipped and leaves the operand unchanged.
func (s *chainVerbScan) argIndex() bool {
	rest := s.format[s.pos:]
	if !strings.HasPrefix(rest, "[") {
		return false
	}
	end := strings.IndexByte(rest, ']')
	if end < 0 {
		s.pos++
		return false
	}
	s.pos += end + 1
	n, err := strconv.Atoi(rest[1:end])
	if err != nil || n < 1 {
		return false
	}
	s.arg = n - 1
	return true
}

// indexedSize reads an optional [n] index and a width or precision after
// it. It reports whether the index still applies to the verb, which it does
// not when a * took the operand it named.
func (s *chainVerbScan) indexedSize() bool {
	indexed := s.argIndex()
	starred := s.sizeOperand()
	return indexed && !starred
}

// skipFlags skips the flag characters of a directive.
func (s *chainVerbScan) skipFlags() {
	for s.pos < len(s.format) && strings.IndexByte("#0+- ", s.format[s.pos]) >= 0 {
		s.pos++
	}
}

// sizeOperand reads a width or precision and reports whether it was a *,
// which takes an operand. Digits take none.
func (s *chainVerbScan) sizeOperand() bool {
	if s.pos < len(s.format) && s.format[s.pos] == '*' {
		s.pos++
		s.arg++
		return true
	}
	for s.pos < len(s.format) && s.format[s.pos] >= '0' && s.format[s.pos] <= '9' {
		s.pos++
	}
	return false
}
