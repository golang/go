// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"

	"cmd/internal/src"
)

// This file holds the loGo/s diagnostic sink (SPEC Sec 16). Lint findings
// and every ErrorfAt or WarnfAt diagnostic print as text, or as one JSON
// object per line when -logos.json is set. print.go carries only the hook
// lines.

// LogosFix is one autofix candidate of a diagnostic.
type LogosFix struct {
	Description      string
	Replacement      string
	InsertBeforeLine int
}

// LogosDiag is a lint finding with its severity already resolved.
type LogosDiag struct {
	// Rule is the rule ID, "group::name".
	Rule string
	// Severity is "forbid", "deny", or "warn". Anything but "warn" fails
	// the compile.
	Severity string
	// Pos and End bound the flagged source; an unknown End means Pos.
	Pos, End src.XPos
	Message  string

	OriginExpr   string
	OriginLine   int
	InferredType string
	Fixes        []LogosFix
}

// logosMsg is the loGo/s part of a queued errorMsg.
type logosMsg struct {
	raw  string     // the message without its position prefix
	warn bool       // a warning, not an error
	diag *LogosDiag // the structured finding, nil for ErrorfAt and WarnfAt
}

// LogosReport queues a lint finding. It counts as an error unless its
// severity is "warn".
func LogosReport(d LogosDiag) {
	text := "[" + d.Rule + "] " + d.Message
	warn := d.Severity == "warn"
	if warn {
		text = "warning: " + text
	}
	msg := text + "\n"
	if d.Pos.IsKnown() {
		msg = FmtPos(d.Pos) + ": " + msg
	}
	errorMsgs = append(errorMsgs, errorMsg{
		pos:   d.Pos,
		msg:   msg,
		logos: logosMsg{raw: d.Message, warn: warn, diag: &d},
	})
	if !warn {
		numErrors++
	}
}

// logosMarkWarn marks the last queued message, just added by WarnfAt, as a
// warning.
func logosMarkWarn() {
	errorMsgs[len(errorMsgs)-1].logos.warn = true
}

// logosText returns the text FlushErrors prints for m: the queued text, or
// its JSON record under -logos.json.
func logosText(m errorMsg) string {
	if !LogosJSON {
		return m.msg
	}
	return logosRecord(m) + "\n"
}

// printTooManyErrors prints the line that ends an error flood: text, or a
// JSON record under -logos.json.
func printTooManyErrors(pos src.XPos) {
	if !LogosJSON {
		fmt.Printf("%v: too many errors\n", FmtPos(pos))
		return
	}
	m := errorMsg{pos: pos, logos: logosMsg{raw: "too many errors"}}
	fmt.Println(logosRecord(m))
}

// jsonSpan is the span object of SPEC Sec 16.3.
type jsonSpan struct {
	StartLine  int `json:"start_line"`
	StartCol   int `json:"start_col"`
	EndLine    int `json:"end_line"`
	EndCol     int `json:"end_col"`
	ByteOffset int `json:"byte_offset"`
	ByteLen    int `json:"byte_len"`
}

// jsonProvenance is the provenance object of SPEC Sec 16.4.
type jsonProvenance struct {
	OriginExpr   string `json:"origin_expr"`
	OriginLine   int    `json:"origin_line"`
	InferredType string `json:"inferred_type"`
}

// jsonFix is an autofix object of SPEC Sec 16.5.
type jsonFix struct {
	Description      string `json:"description"`
	Replacement      string `json:"replacement"`
	InsertBeforeLine int    `json:"insert_before_line,omitzero"`
}

// jsonRecord is one diagnostic line of SPEC Sec 16.2.
type jsonRecord struct {
	SchemaVersion     string         `json:"schema_version"`
	Severity          string         `json:"severity"`
	RuleID            string         `json:"rule_id"`
	File              string         `json:"file"`
	Span              jsonSpan       `json:"span"`
	Message           string         `json:"message"`
	Provenance        jsonProvenance `json:"provenance"`
	AutofixCandidates []jsonFix      `json:"autofix_candidates"`
}

// ruleTag matches the "[group::name] " prefix of a diagnostic that names its
// rule.
var ruleTag = regexp.MustCompile(`^\[([a-z]+::[a-z_]+)\] `)

// logosRecord renders m as one JSON object. It uses encoding/json, not
// encoding/json/v2: the bootstrap toolchain builds this package, and v2 is
// an experiment there.
func logosRecord(m errorMsg) string {
	rec := jsonRecord{
		SchemaVersion:     "1.0",
		AutofixCandidates: []jsonFix{},
	}
	if d := m.logos.diag; d != nil {
		rec.RuleID, rec.Severity, rec.Message = d.Rule, d.Severity, m.logos.raw
		rec.Provenance = jsonProvenance{OriginExpr: d.OriginExpr, OriginLine: d.OriginLine, InferredType: d.InferredType}
		for _, f := range d.Fixes {
			rec.AutofixCandidates = append(rec.AutofixCandidates, jsonFix(f))
		}
		rec.Span = logosSpan(d.Pos, d.End)
		rec.File = logosFile(d.Pos)
	} else {
		rec.RuleID, rec.Message = splitRuleTag(m.logos.raw)
		rec.Severity = compilerSeverity(rec.RuleID)
		if m.logos.warn {
			rec.Severity = "warn"
		}
		rec.Span = logosSpan(m.pos, m.pos)
		rec.File = logosFile(m.pos)
	}
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(rec); err != nil {
		return fmt.Sprintf(`{"schema_version":"1.0","severity":"deny","rule_id":"","file":"","message":%q}`, err.Error())
	}
	return strings.TrimSuffix(out.String(), "\n")
}

// splitRuleTag splits a leading "[group::name] " off msg.
func splitRuleTag(msg string) (rule, rest string) {
	m := ruleTag.FindStringSubmatch(msg)
	if m == nil {
		return "", msg
	}
	return m[1], msg[len(m[0]):]
}

// compilerSeverity returns the severity of a compiler error: forbid for an
// error of a correctness or concurrency rule, and deny for every other
// error, including each Go compiler error.
func compilerSeverity(rule string) string {
	if strings.HasPrefix(rule, "correctness::") || strings.HasPrefix(rule, "concurrency::") {
		return "forbid"
	}
	return "deny"
}

// logosFile returns the file name text diagnostics print for pos.
func logosFile(pos src.XPos) string {
	if !pos.IsKnown() || Ctxt == nil {
		return ""
	}
	return Ctxt.OutermostPos(pos).RelFilename()
}

// logosSpan returns the span of pos through end. Lines and columns match
// the text diagnostic; the byte range is measured in the file on disk, and
// is zero when the file cannot be read.
func logosSpan(pos, end src.XPos) jsonSpan {
	if !pos.IsKnown() || Ctxt == nil {
		return jsonSpan{}
	}
	if !end.IsKnown() || end.Before(pos) {
		end = pos
	}
	p, e := Ctxt.OutermostPos(pos), Ctxt.OutermostPos(end)
	s := jsonSpan{
		StartLine: int(p.RelLine()), StartCol: int(p.RelCol()),
		EndLine: int(e.RelLine()), EndCol: int(e.RelCol()),
	}
	start, ok1 := byteOffset(p.Filename(), p.Line(), p.Col())
	stop, ok2 := byteOffset(e.Filename(), e.Line(), e.Col())
	if ok1 && ok2 {
		s.ByteOffset, s.ByteLen = start, max(0, stop-start)
	}
	return s
}

// lineStarts caches the byte offset of each line start of the files named
// in diagnostics.
var lineStarts = make(map[string][]int)

// byteOffset returns the byte offset of line, col (both 1-based) in the
// named file.
func byteOffset(name string, line, col uint) (int, bool) {
	starts, ok := lineStarts[name]
	if !ok {
		starts = readLineStarts(name)
		lineStarts[name] = starts
	}
	if line == 0 || int(line) > len(starts) || col == 0 {
		return 0, false
	}
	return starts[line-1] + int(col) - 1, true
}

// readLineStarts returns the offset of every line start of the named file,
// or nil when it cannot be read.
func readLineStarts(name string) []int {
	data, err := os.ReadFile(name)
	if err != nil {
		return nil
	}
	starts := []int{0}
	for i, b := range data {
		if b == '\n' {
			starts = append(starts, i+1)
		}
	}
	return starts
}
