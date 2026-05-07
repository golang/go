// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgutil

import (
	"bytes"
	"io"
	"text/template"
)

type TforAsBits interface {
	ElemBits() int
	Name_() string
}

const (
	ToBitsName        = "ToBits"
	ToIntsName        = "BitsToInt{{.To.ElemBits}}"
	ConvertToIntName  = "ConvertToInt{{.To.ElemBits}}"
	ConvertToUintName = "ConvertToUint{{.From.ElemBits}}" // backwards because done with ToInts
	ToFloatsName      = "BitsToFloat{{.To.ElemBits}}"
	ReshapeName       = "ReshapeToUint{{.To.ElemBits}}s"
)

func templateOf(name, text string) *template.Template {
	return template.Must(template.New(name).Parse(text))
}

var ToBitsCall = templateOf("toBitsCall", "."+ToBitsName+"()")
var ToIntsCall = templateOf("toBitsCall", "."+ToIntsName+"()")
var ToFloatsCall = templateOf("toBitsCall", "."+ToFloatsName+"()")
var ReshapeCall = templateOf("toBitsCall", "."+ReshapeName+"()")

var ToBitsDcl = templateOf("toBitsDcl", `
	// `+ToBitsName+` reinterprets the bits of a {{.From.Name_}} vector as a {{.To.Name_}} vector
	func (x {{.From.Name_}}) `+ToBitsName+`() {{.To.Name_}}
`)

// ToBitsIntrinsic is used in a uint -> T context, but reverse, so T is method receiver.
var ToBitsIntrinsic = templateOf("toBitsIntrin", `{{.To.Name_}}.`+ToBitsName)

var ToIntsDcl = templateOf("toIntsDcl", `
	// `+ToIntsName+` reinterprets the bits of a {{.From.Name_}} vector as a {{.To.Name_}} vector
	func (x {{.From.Name_}}) `+ToIntsName+`() {{.To.Name_}}

	// `+ConvertToIntName+` converts a {{.From.Name_}} vector to a {{.To.Name_}} vector
	func (x {{.From.Name_}}) `+ConvertToIntName+`() {{.To.Name_}}

	// `+ConvertToUintName+` converts a {{.To.Name_}} vector to a {{.From.Name_}} vector
	func (x {{.To.Name_}}) `+ConvertToUintName+`() {{.From.Name_}}
`)

var ToIntsIntrinsic = templateOf("toIntsIntrin", `{{.From.Name_}}.`+ToIntsName)
var CvtToIntsIntrinsic = templateOf("cvtToIntIntrin", `{{.From.Name_}}.`+ConvertToIntName)

// Also used in a to-from-reversed context.
var CvtToUintsIntrinsic = templateOf("cvtToUintIntrin", `{{.To.Name_}}.`+ConvertToUintName)

var ToFloatsDcl = templateOf("toFloatsDcl", `
	// `+ToFloatsName+` reinterprets the bits of a {{.From.Name_}} vector as a {{.To.Name_}} vector
	func (x {{.From.Name_}}) `+ToFloatsName+`() {{.To.Name_}}
`)

var ToFloatsIntrinsic = templateOf("toFloatsIntrin", `{{.From.Name_}}.`+ToFloatsName)

var ReshapeDcl = templateOf("reshapeDcl", `
	// `+ReshapeName+` reinterprets the bits of a {{.From.Name_}} vector as a {{.To.Name_}} vector
	func (x {{.From.Name_}}) `+ReshapeName+`() {{.To.Name_}}
`)

var ReshapeIntrinsic = templateOf("reshapeIntrinsic", `{{.From.Name_}}.`+ReshapeName)

var AsOp = templateOf("asConversion", `
	// As{{.To.Name}} reinterprets the bits of a {{.From.Name}} vector as a {{.To.Name}} vector
	//
	// Deprecated: use combinations of ToBits, BitsTo{Int<N>,Float<N>}, ReshapeToUint<N>
	//
	//go:fix inline
	func (x {{.From.Name}}) As{{.To.Name}}() {{.To.Name}} {
		return x{{.AsTranslation}}
	}
`)

type AsConversion struct {
	From, To TforAsBits
}

func Conversion(from, to TforAsBits) *AsConversion {
	return &AsConversion{from, to}
}

type Foo struct {
	Foo string
}

func (c *AsConversion) Foo(t *template.Template) *Foo {
	var b bytes.Buffer
	t.Execute(&b, c)
	return &Foo{b.String()}
}

func (c *AsConversion) ExecuteIntrinsicTemplateOfFoo(w io.Writer, t *template.Template) {
	from, to := c.From, c.To
	switch to.Name_()[0] {
	case 'F': // U -> F
		t.Execute(w, c.Foo(ToFloatsIntrinsic))
		t.Execute(w, c.Foo(ToBitsIntrinsic))
	case 'I': // U -> I
		t.Execute(w, c.Foo(ToIntsIntrinsic))
		t.Execute(w, c.Foo(CvtToIntsIntrinsic))
		t.Execute(w, c.Foo(CvtToUintsIntrinsic))
		t.Execute(w, c.Foo(ToBitsIntrinsic))
	case 'U': // U -> U
		if from.Name_()[0] == 'U' {
			t.Execute(w, c.Foo(ReshapeIntrinsic))
		}
	}
}

func (a *AsConversion) AsTranslation() string {
	var b bytes.Buffer
	if a.From.Name_()[0] != 'U' {
		ToBitsCall.Execute(&b, a)
	}
	if a.From.ElemBits() != a.To.ElemBits() {
		ReshapeCall.Execute(&b, a)
	}
	if a.To.Name_()[0] == 'F' {
		ToFloatsCall.Execute(&b, a)
	}
	if a.To.Name_()[0] == 'I' {
		ToIntsCall.Execute(&b, a)
	}
	return b.String()
}
