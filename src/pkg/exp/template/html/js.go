// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"fmt"
	"json"
	"strings"
	"utf8"
)

// nextJSCtx returns the context that determines whether a slash after the
// given run of tokens tokens starts a regular expression instead of a division
// operator: / or /=.
//
// This assumes that the token run does not include any string tokens, comment
// tokens, regular expression literal tokens, or division operators.
//
// This fails on some valid but nonsensical JavaScript programs like
// "x = ++/foo/i" which is quite different than "x++/foo/i", but is not known to
// fail on any known useful programs. It is based on the draft
// JavaScript 2.0 lexical grammar and requires one token of lookbehind:
// http://www.mozilla.org/js/language/js20-2000-07/rationale/syntax.html
func nextJSCtx(s []byte, preceding jsCtx) jsCtx {
	s = bytes.TrimRight(s, "\t\n\f\r \u2028\u2029")
	if len(s) == 0 {
		return preceding
	}

	// All cases below are in the single-byte UTF-8 group.
	switch c, n := s[len(s)-1], len(s); c {
	case '+', '-':
		// ++ and -- are not regexp preceders, but + and - are whether
		// they are used as infix or prefix operators.
		start := n - 1
		// Count the number of adjacent dashes or pluses.
		for start > 0 && s[start-1] == c {
			start--
		}
		if (n-start)&1 == 1 {
			// Reached for trailing minus signs since "---" is the
			// same as "-- -".
			return jsCtxRegexp
		}
		return jsCtxDivOp
	case '.':
		// Handle "42."
		if n != 1 && '0' <= s[n-2] && s[n-2] <= '9' {
			return jsCtxDivOp
		}
		return jsCtxRegexp
	// Suffixes for all punctuators from section 7.7 of the language spec
	// that only end binary operators not handled above.
	case ',', '<', '>', '=', '*', '%', '&', '|', '^', '?':
		return jsCtxRegexp
	// Suffixes for all punctuators from section 7.7 of the language spec
	// that are prefix operators not handled above.
	case '!', '~':
		return jsCtxRegexp
	// Matches all the punctuators from section 7.7 of the language spec
	// that are open brackets not handled above.
	case '(', '[':
		return jsCtxRegexp
	// Matches all the punctuators from section 7.7 of the language spec
	// that precede expression starts.
	case ':', ';', '{':
		return jsCtxRegexp
	// CAVEAT: the close punctuators ('}', ']', ')') precede div ops and
	// are handled in the default except for '}' which can precede a
	// division op as in
	//    ({ valueOf: function () { return 42 } } / 2
	// which is valid, but, in practice, developers don't divide object
	// literals, so our heuristic works well for code like
	//    function () { ... }  /foo/.test(x) && sideEffect();
	// The ')' punctuator can precede a regular expression as in
	//     if (b) /foo/.test(x) && ...
	// but this is much less likely than
	//     (a + b) / c
	case '}':
		return jsCtxRegexp
	default:
		// Look for an IdentifierName and see if it is a keyword that
		// can precede a regular expression.
		j := n
		for j > 0 && isJSIdentPart(int(s[j-1])) {
			j--
		}
		if regexpPrecederKeywords[string(s[j:])] {
			return jsCtxRegexp
		}
	}
	// Otherwise is a punctuator not listed above, or
	// a string which precedes a div op, or an identifier
	// which precedes a div op.
	return jsCtxDivOp
}

// regexPrecederKeywords is a set of reserved JS keywords that can precede a
// regular expression in JS source.
var regexpPrecederKeywords = map[string]bool{
	"break":      true,
	"case":       true,
	"continue":   true,
	"delete":     true,
	"do":         true,
	"else":       true,
	"finally":    true,
	"in":         true,
	"instanceof": true,
	"return":     true,
	"throw":      true,
	"try":        true,
	"typeof":     true,
	"void":       true,
}

// jsValEscaper escapes its inputs to a JS Expression (section 11.14) that has
// nether side-effects nor free variables outside (NaN, Infinity).
func jsValEscaper(args ...interface{}) string {
	var a interface{}
	if len(args) == 1 {
		a = args[0]
	} else {
		a = fmt.Sprint(args...)
	}
	// TODO: detect cycles before calling Marshal which loops infinitely on
	// cyclic data. This may be an unnacceptable DoS risk.

	// TODO: make sure that json.Marshal escapes codepoints U+2028 & U+2029
	// so it falls within the subset of JSON which is valid JS and maybe
	// post-process to prevent it from containing
	// "<!--", "-->", "<![CDATA[", "]]>", or "</script"
	// in case custom marshallers produce output containing those.

	// TODO: Maybe abbreviate \u00ab to \xab to produce more compact output.

	// TODO: JSON allows arbitrary unicode codepoints, but EcmaScript
	// defines a SourceCharacter as either a UTF-16 or UCS-2 code-unit.
	// Determine whether supplemental codepoints in UTF-8 encoded JS inside
	// string literals are properly interpreted by major interpreters.

	b, err := json.Marshal(a)
	if err != nil {
		// Put a space before comment so that if it is flush against
		// a division operator it is not turned into a line comment:
		//     x/{{y}}
		// turning into
		//     x//* error marshalling y:
		//          second line of error message */null
		return fmt.Sprintf(" /* %s */null ", strings.Replace(err.String(), "*/", "* /", -1))
	}
	if len(b) != 0 {
		first, _ := utf8.DecodeRune(b)
		last, _ := utf8.DecodeLastRune(b)
		if isJSIdentPart(first) || isJSIdentPart(last) {
			return " " + string(b) + " "
		}
	}
	return string(b)
}

// jsStrEscaper produces a string that can be included between quotes in
// JavaScript source, in JavaScript embedded in an HTML5 <script> element,
// or in an HTML5 event handler attribute such as onclick.
func jsStrEscaper(args ...interface{}) string {
	return replace(stringify(args...), jsStrReplacementTable)
}

// jsRegexpEscaper behaves like jsStrEscaper but escapes regular expression
// specials so the result is treated literally when included in a regular
// expression literal. /foo{{.X}}bar/ matches the string "foo" followed by
// the literal text of {{.X}} followed by the string "bar".
func jsRegexpEscaper(args ...interface{}) string {
	s := replace(stringify(args...), jsRegexpReplacementTable)
	if s == "" {
		// /{{.X}}/ should not produce a line comment when .X == "".
		return "(?:)"
	}
	return s
}

// stringify is an optimized form of fmt.Sprint.
func stringify(args ...interface{}) string {
	if len(args) == 1 {
		if s, ok := args[0].(string); ok {
			return s
		}
	}
	return fmt.Sprint(args...)
}

// replace replaces each rune r of s with replacementTable[r], provided that
// r < len(replacementTable). If replacementTable[r] is the empty string then
// no replacement is made.
// It also replaces the runes '\u2028' and '\u2029' with the strings
// `\u2028` and `\u2029`. Note the different quotes used.
func replace(s string, replacementTable []string) string {
	var b bytes.Buffer
	written := 0
	for i, r := range s {
		var repl string
		switch {
		case r < len(replacementTable) && replacementTable[r] != "":
			repl = replacementTable[r]
		case r == '\u2028':
			repl = `\u2028`
		case r == '\u2029':
			repl = `\u2029`
		default:
			continue
		}
		b.WriteString(s[written:i])
		b.WriteString(repl)
		written = i + utf8.RuneLen(r)
	}
	if written == 0 {
		return s
	}
	b.WriteString(s[written:])
	return b.String()
}

var jsStrReplacementTable = []string{
	0:    `\0`,
	'\t': `\t`,
	'\n': `\n`,
	'\v': `\x0b`, // "\v" == "v" on IE 6.
	'\f': `\f`,
	'\r': `\r`,
	// Encode HTML specials as hex so the output can be embedded
	// in HTML attributes without further encoding.
	'"':  `\x22`,
	'&':  `\x26`,
	'\'': `\x27`,
	'+':  `\x2b`,
	'/':  `\/`,
	'<':  `\x3c`,
	'>':  `\x3e`,
	'\\': `\\`,
}

var jsRegexpReplacementTable = []string{
	0:    `\0`,
	'\t': `\t`,
	'\n': `\n`,
	'\v': `\x0b`, // "\v" == "v" on IE 6.
	'\f': `\f`,
	'\r': `\r`,
	// Encode HTML specials as hex so the output can be embedded
	// in HTML attributes without further encoding.
	'"':  `\x22`,
	'$':  `\$`,
	'&':  `\x26`,
	'\'': `\x27`,
	'(':  `\(`,
	')':  `\)`,
	'*':  `\*`,
	'+':  `\x2b`,
	'-':  `\-`,
	'.':  `\.`,
	'/':  `\/`,
	'<':  `\x3c`,
	'>':  `\x3e`,
	'?':  `\?`,
	'[':  `\[`,
	'\\': `\\`,
	']':  `\]`,
	'^':  `\^`,
	'{':  `\{`,
	'|':  `\|`,
	'}':  `\}`,
}

// isJSIdentPart returns whether the given rune is a JS identifier part.
// It does not handle all the non-Latin letters, joiners, and combining marks,
// but it does handle every codepoint that can occur in a numeric literal or
// a keyword.
func isJSIdentPart(rune int) bool {
	switch {
	case '$' == rune:
		return true
	case '0' <= rune && rune <= '9':
		return true
	case 'A' <= rune && rune <= 'Z':
		return true
	case '_' == rune:
		return true
	case 'a' <= rune && rune <= 'z':
		return true
	}
	return false
}
