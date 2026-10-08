// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sgutil provides shared utilities for SIMD file
// generation across architectures.  This includes
//
// - "natural" comparison for better ordering
// - formatted-Go file saving
// - generic ops file generation
// - naming conventions and templates for the
//   bitwise vector reinterpretation no-op methods.

package sgutil

import (
	"fmt"
	"io"
	"sort"
	"text/template"
)

const simdGenericOpsTmpl = `package main

func init() {
	additionalGenericOps["{{.ArchKey}}"] = []opData{
{{- range .Ops }}
		{name: "{{.OpName}}", argLength: {{.OpInLen}}{{if .Comm}}, commutative: true{{end}}{{if .HasAux}}, aux: "UInt8"{{end}}},
{{- end }}
	}
}
`

// TemplateNamed returns a parsed template from temp, named name.
func TemplateNamed(name, temp string) *template.Template {
	t, err := template.New(name).Parse(temp)
	if err != nil {
		panic(fmt.Errorf("failed to parse template %s: %w", name, err))
	}
	return t
}

// GenericOpsData holds one generic op entry for template rendering.
type GenericOpsData struct {
	OpName  string
	OpInLen int
	Comm    bool
	HasAux  bool
}

// WriteSIMDGenericOps generates the generic ops file content for archKey.
func WriteSIMDGenericOps(w io.Writer, ops []GenericOpsData, archKey string) {
	sort.Slice(ops, func(i, j int) bool {
		return CompareNatural(ops[i].OpName, ops[j].OpName) < 0
	})

	var deduped []GenericOpsData
	for _, op := range ops {
		if len(deduped) > 0 && deduped[len(deduped)-1].OpName == op.OpName {
			if deduped[len(deduped)-1].OpInLen != op.OpInLen ||
				deduped[len(deduped)-1].Comm != op.Comm ||
				deduped[len(deduped)-1].HasAux != op.HasAux {
				panic(fmt.Sprintf("conflicting generic op definition for %s", op.OpName))
			}
			continue
		}
		deduped = append(deduped, op)
	}

	type templateData struct {
		Header  string
		ArchKey string
		Ops     []GenericOpsData
	}

	t := TemplateNamed("simdgenericOps", simdGenericOpsTmpl)
	err := t.Execute(w, templateData{
		ArchKey: archKey,
		Ops:     deduped,
	})
	if err != nil {
		panic(fmt.Errorf("failed to execute template: %w", err))
	}
}
