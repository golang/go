// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"regexp"
	"sync"
	"text/template"
)

// Presentation generates output from a corpus.
type Presentation struct {
	Corpus *Corpus

	DirlistHTML,
	ErrorHTML,
	ExampleHTML,
	GodocHTML,
	PackageHTML,
	PackageText,
	SearchHTML,
	SearchText,
	SearchDescXML *template.Template

	// TabWidth optionally specifies the tab width.
	TabWidth int

	ShowTimestamps bool
	ShowPlayground bool
	ShowExamples   bool
	DeclLinks      bool

	// NotesRx optionally specifies a regexp to match
	// notes to render in the output.
	NotesRx *regexp.Regexp

	initFuncMapOnce sync.Once
	funcMap         template.FuncMap
	templateFuncs   template.FuncMap
}

// NewPresentation returns a new Presentation from a corpus.
func NewPresentation(c *Corpus) *Presentation {
	if c == nil {
		panic("nil Corpus")
	}
	return &Presentation{
		Corpus:       c,
		TabWidth:     4,
		ShowExamples: true,
		DeclLinks:    true,
	}
}
