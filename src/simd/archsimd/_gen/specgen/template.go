// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"bytes"
	"cmp"
	"fmt"
	"reflect"
	"regexp"
	"simd/archsimd/_gen/specgen/specexpr"
	"strings"
	"text/template"
)

// specTemplate wraps a parsed text/template.Template, or a plain string if no
// template actions were present.
type specTemplate struct {
	raw  string
	tmpl *template.Template
}

var templateFuncs = template.FuncMap{
	"title": func(v any) string {
		s := fmt.Sprint(v)
		if len(s) == 0 {
			return ""
		}
		return strings.ToTitle(s[:1]) + s[1:]
	},
	"lt": func(a, b any) (bool, error) {
		res, ok := compare(a, b)
		if !ok {
			return false, fmt.Errorf("incomparable types: %T and %T", a, b)
		}
		return res < 0, nil
	},
	"le": func(a, b any) (bool, error) {
		res, ok := compare(a, b)
		if !ok {
			return false, fmt.Errorf("incomparable types: %T and %T", a, b)
		}
		return res <= 0, nil
	},
	"gt": func(a, b any) (bool, error) {
		res, ok := compare(a, b)
		if !ok {
			return false, fmt.Errorf("incomparable types: %T and %T", a, b)
		}
		return res > 0, nil
	},
	"ge": func(a, b any) (bool, error) {
		res, ok := compare(a, b)
		if !ok {
			return false, fmt.Errorf("incomparable types: %T and %T", a, b)
		}
		return res >= 0, nil
	},
	"eq": func(a, b any) bool {
		if res, ok := compare(a, b); ok {
			return res == 0
		}
		return reflect.DeepEqual(a, b)
	},
	"ne": func(a, b any) bool {
		if res, ok := compare(a, b); ok {
			return res != 0
		}
		return !reflect.DeepEqual(a, b)
	},
}

func toSpecNum(v any) (specexpr.Num, bool) {
	switch v := v.(type) {
	case specexpr.Num:
		return v, true
	case int:
		return specexpr.Int(v), true
	case int64:
		return specexpr.Int(v), true
	default:
		return nil, false
	}
}

func compare(a, b any) (int, bool) {
	if na, okA := toSpecNum(a); okA {
		if nb, okB := toSpecNum(b); okB {
			return na.Compare(nb)
		}
	}
	if sa, okA := a.(string); okA {
		if sb, okB := b.(string); okB {
			return cmp.Compare(sa, sb), true
		}
	}
	return 0, false
}

// newSpecTemplate parses a spec template string. If tmpl does not contain "{{",
// it is treated as a raw string literal without template overhead.
func newSpecTemplate(tmpl string) (specTemplate, error) {
	if !strings.Contains(tmpl, "{{") {
		return specTemplate{raw: tmpl, tmpl: nil}, nil
	}

	t, err := template.New("").Option("missingkey=error").Funcs(templateFuncs).Parse(tmpl)
	if err != nil {
		return specTemplate{}, err
	}
	return specTemplate{raw: tmpl, tmpl: t}, nil
}

func (s *specTemplate) expand(data any) (string, error) {
	if s.tmpl == nil {
		return s.raw, nil
	}
	var buf bytes.Buffer
	if err := s.tmpl.Execute(&buf, data); err != nil {
		return "", err
	}
	return buf.String(), nil
}

var reMultipleNewlines = regexp.MustCompile(`\n{3,}`)

func cleanDocNewlines(s string) string {
	s = reMultipleNewlines.ReplaceAllString(s, "\n\n")
	trimmed := strings.TrimRight(s, " \t\n")
	if trimmed == "" {
		return ""
	}
	return trimmed + "\n"
}

// expandNameAndDoc instantiates the API function name and doc comment for sFn
// using the solved variable bindings in b.
func (sFn *specFunc) expandNameAndDoc(ctx context, b *specexpr.Bindings) (name, doc string) {
	data := make(map[string]any)
	for v, val := range b.All() {
		data[string(v)] = val
		if s, ok := strings.CutPrefix(string(v), "$"); ok {
			data[s] = val
		}
	}

	var err error
	name, err = sFn.NameTmpl.expand(data)
	if err != nil {
		ctx.errorf("expanding function name template: %s", err)
		name = sFn.Name
	}

	data["Name"] = name
	doc, err = sFn.Doc.expand(data)
	if err != nil {
		ctx.errorf("expanding doc template: %s", err)
		doc = sFn.Doc.raw
	}

	doc = cleanDocNewlines(doc)

	if name != sFn.Name {
		doc = regexp.MustCompile(`\b`+regexp.QuoteMeta(sFn.Name)+`\b`).ReplaceAllLiteralString(doc, name)
	}

	return name, doc
}
