// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"strings"
	"testing"
)

const typedNilLgoSrc = `package p

import (
	"io"
	"os"
)

type Err struct{ msg string }

func (e *Err) Error() string { return e.msg }

func conversion() error {
	return (*Err)(nil) // want
}

func conversionParen() any {
	return ((*Err)(nil)) // want
}

func conversionSlot() (int, error) {
	return 1, (*Err)(nil) // want
}

func conversionToPointer() (*Err?, error) {
	return (*Err)(nil), nil
}

func option(p *Err?) any {
	return p // want
}

func optionProved(p *Err?) any {
	if p == nil {
		return nil
	}
	return p
}

func optionGuarded(p *Err?) any {
	if p != nil {
		return p
	}
	return nil
}

func optionOther(p, q *Err?) any {
	if q != nil {
		return p // want
	}
	return nil
}

func optionReassigned(p, q *Err?) any {
	if p == nil {
		return nil
	}
	p = q
	return p // want
}

func optionClosure(p *Err?) func() any {
	return func() any {
		return p // want
	}
}

func platform() io.Closer {
	f := os.Stdout
	return f // want
}

func platformProved() io.Closer {
	f := os.Stdout
	if f == nil {
		return nil
	}
	return f
}

func platformTuple(name string) (io.ReadCloser, error) {
	return os.Open(name) // want
}

func platformConcrete() *os.File {
	return os.Stdout
}

func platformTupleBoxedSlot() (*os.File, io.Writer, error) {
	return os.Pipe() // want
}

func platformTupleConcreteSlots(name string) (*os.File, any) {
	return os.Open(name)
}

func platformTupleErrorBoxed() (*os.File, *os.File, any) {
	return os.Pipe()
}

func nonNil(e *Err) error {
	return e
}

func optionNestedBlock(p *Err?) any {
	{
		if p == nil {
			return nil
		}
	}
	return p
}

func optionNestedBlockKilled(p *Err?, q *Err?) any {
	{
		if p == nil {
			return nil
		}
		p = q
	}
	return p // want
}
`

const typedNilGoSrc = `package p

import (
	"io"
	"os"
)

type Err struct{ msg string }

func (e *Err) Error() string { return e.msg }

func conversion() error {
	return (*Err)(nil) // want
}

func conversionCloser() io.Closer {
	return (*os.File)(nil) // want
}

func conversionClosure() func() error {
	return func() error {
		return (*Err)(nil) // want
	}
}

func conversionConcrete() *Err {
	return (*Err)(nil)
}

func variable(p *Err) error {
	return p
}

func platform() io.Closer {
	return os.Stdout
}

func sliceConversion() any {
	return []int(nil)
}
`

// TestTypedNilReturn covers the three forms of correctness::typed_nil_return
// in a loGo/s package, and the (*T)(nil) form alone in a Go package.
func TestTypedNilReturn(t *testing.T) {
	expect(t, KindLgo, nil, testSource{"x.lgo", typedNilLgoSrc}, finding{"correctness::typed_nil_return", Forbid})
	expect(t, KindGo, nil, testSource{"x.go", typedNilGoSrc}, finding{"correctness::typed_nil_return", Deny})
}

// TestTypedNilReturnMessage checks the text and the span of each form: the
// span covers the returned expression, through the closing parenthesis of a
// call.
func TestTypedNilReturnMessage(t *testing.T) {
	in := check(t, KindLgo, "x.lgo", typedNilLgoSrc)
	var texts []string
	for _, r := range Run(in) {
		texts = append(texts, r.Message+" | "+spanText(typedNilLgoSrc, r))
	}
	for _, want := range []string{
		"(*Err)(nil) returned as error result is a typed nil: the interface is not nil; return nil instead | (*Err)(nil)",
		"(*Err)(nil) returned as any result is a typed nil: the interface is not nil; return nil instead | (*Err)(nil)",
		"option p (*Err?) returned as any result without a dominating nil check: a none value boxes into a non-nil interface | p",
		"platform pointer f (*os.File) returned as io.Closer result without a dominating nil check: a nil pointer boxes into a non-nil interface | f",
		"platform pointer os.Open(name) (*os.File) returned as io.ReadCloser result without a dominating nil check: a nil pointer boxes into a non-nil interface | os.Open(name)",
	} {
		if !strings.Contains(strings.Join(texts, "\n"), want) {
			t.Errorf("missing finding %q in\n%s", want, strings.Join(texts, "\n"))
		}
	}
}

// TestTypedNilReturnForbid checks that a //logos:allow directive silences the
// rule in a Go package, where the Go cap lowers forbid to deny, and does not
// in a loGo/s package.
func TestTypedNilReturnForbid(t *testing.T) {
	const goSrc = `package p

type Err struct{}

func (*Err) Error() string { return "" }

//logos:allow(correctness::typed_nil_return)
func f() error {
	return (*Err)(nil)
}
`
	if got := summarize(Run(check(t, KindGo, "x.go", goSrc))); len(got) != 0 {
		t.Errorf("Go package results %q, want none", got)
	}
	got := summarize(Run(check(t, KindLgo, "x.lgo", goSrc)))
	want := []string{
		"7 correctness::allow_forbidden_rule forbid",
		"9 correctness::typed_nil_return forbid",
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("loGo/s package results %q, want %q", got, want)
	}
}
