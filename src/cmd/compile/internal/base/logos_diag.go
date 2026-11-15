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

type LogosFix struct {
	Description      string
	Replacement      string
	InsertBeforeLine int
	Note             string
}

type LogosDiag struct {
	Rule     string
	Severity string
	Pos, End src.XPos
	Message  string
	Note     string

	OriginExpr   string
	OriginLine   int
	InferredType string
	Fixes        []LogosFix
}

type logosMsg struct {
	raw  string
	warn bool
	diag *LogosDiag
}

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

func logosMarkWarn() {
	errorMsgs[len(errorMsgs)-1].logos.warn = true
}

func logosText(m errorMsg) string {
	if !LogosJSON {
		return m.msg
	}
	return logosRecord(m) + "\n"
}

func printTooManyErrors(pos src.XPos) {
	if !LogosJSON {
		fmt.Printf("%v: too many errors\n", FmtPos(pos))
		return
	}
	m := errorMsg{pos: pos, logos: logosMsg{raw: "too many errors"}}
	fmt.Println(logosRecord(m))
}

type jsonSpan struct {
	StartLine  int `json:"start_line"`
	StartCol   int `json:"start_col"`
	EndLine    int `json:"end_line"`
	EndCol     int `json:"end_col"`
	ByteOffset int `json:"byte_offset"`
	ByteLen    int `json:"byte_len"`
}

type jsonProvenance struct {
	OriginExpr   string `json:"origin_expr"`
	OriginLine   int    `json:"origin_line"`
	InferredType string `json:"inferred_type"`
}

type jsonFix struct {
	Description      string `json:"description"`
	Replacement      string `json:"replacement"`
	InsertBeforeLine int    `json:"insert_before_line,omitzero"`
	Note             string `json:"note,omitzero"`
}

type jsonRecord struct {
	SchemaVersion     string         `json:"schema_version"`
	Severity          string         `json:"severity"`
	RuleID            string         `json:"rule_id"`
	File              string         `json:"file"`
	Span              jsonSpan       `json:"span"`
	Message           string         `json:"message"`
	Note              string         `json:"note,omitzero"`
	Provenance        jsonProvenance `json:"provenance"`
	AutofixCandidates []jsonFix      `json:"autofix_candidates"`
}

var ruleTag = regexp.MustCompile(`^\[([a-z]+::[a-z_]+)\] `)

func logosRecord(m errorMsg) string {
	rec := jsonRecord{
		SchemaVersion:     "1.0",
		AutofixCandidates: []jsonFix{},
	}
	if d := m.logos.diag; d != nil {
		rec.RuleID, rec.Severity, rec.Message = d.Rule, d.Severity, m.logos.raw
		rec.Note = d.Note
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

func splitRuleTag(msg string) (rule, rest string) {
	m := ruleTag.FindStringSubmatch(msg)
	if m == nil {
		return "", msg
	}
	return m[1], msg[len(m[0]):]
}

func compilerSeverity(rule string) string {
	if strings.HasPrefix(rule, "correctness::") || strings.HasPrefix(rule, "concurrency::") {
		return "forbid"
	}
	return "deny"
}

func logosFile(pos src.XPos) string {
	if !pos.IsKnown() || Ctxt == nil {
		return ""
	}
	return Ctxt.OutermostPos(pos).RelFilename()
}

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

var lineStarts = make(map[string][]int)

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
