// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dnsmessage

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func mustNewName(name string) Name {
	n, err := NewName(name)
	if err != nil {
		panic(err)
	}
	return n
}

func (m *Message) String() string {
	s := fmt.Sprintf("Message: %#v\n", &m.Header)
	if len(m.Questions) > 0 {
		s += "-- Questions\n"
		for _, q := range m.Questions {
			s += fmt.Sprintf("%#v\n", q)
		}
	}
	if len(m.Answers) > 0 {
		s += "-- Answers\n"
		for _, a := range m.Answers {
			s += fmt.Sprintf("%#v\n", a)
		}
	}
	if len(m.Authorities) > 0 {
		s += "-- Authorities\n"
		for _, ns := range m.Authorities {
			s += fmt.Sprintf("%#v\n", ns)
		}
	}
	if len(m.Additionals) > 0 {
		s += "-- Additionals\n"
		for _, e := range m.Additionals {
			s += fmt.Sprintf("%#v\n", e)
		}
	}
	return s
}

func TestNameString(t *testing.T) {
	want := "foo"
	name := mustNewName(want)
	if got := fmt.Sprint(name); got != want {
		t.Errorf("got fmt.Sprint(%#v) = %s, want = %s", name, got, want)
	}
}

func TestQuestionPackUnpack(t *testing.T) {
	want := Question{
		Name:  mustNewName("."),
		Type:  TypeA,
		Class: ClassINET,
	}
	buf, err := want.pack(make([]byte, 1, 50), map[string]int{}, 1)
	if err != nil {
		t.Fatal("Packing failed:", err)
	}
	var p Parser
	p.msg = buf
	p.header.questions = 1
	p.section = sectionQuestions
	p.off = 1
	got, err := p.Question()
	if err != nil {
		t.Fatalf("Unpacking failed: %v\n%s", err, string(buf[1:]))
	}
	if p.off != len(buf) {
		t.Errorf("Unpacked different amount than packed: got n = %d, want = %d", p.off, len(buf))
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Got = %+v, want = %+v", got, want)
	}
}

func TestName(t *testing.T) {
	tests := []string{
		"",
		".",
		"google..com",
		"google.com",
		"google..com.",
		"google.com.",
		".google.com.",
		"www..google.com.",
		"www.google.com.",
	}

	for _, test := range tests {
		n, err := NewName(test)
		if err != nil {
			t.Errorf("Creating name for %q: %v", test, err)
			continue
		}
		if ns := n.String(); ns != test {
			t.Errorf("Got %#v.String() = %q, want = %q", n, ns, test)
			continue
		}
	}
}

func TestNamePackUnpack(t *testing.T) {
	tests := []struct {
		in   string
		want string
		err  error
	}{
		{"", "", errNonCanonicalName},
		{".", ".", nil},
		{"google..com", "", errNonCanonicalName},
		{"google.com", "", errNonCanonicalName},
		{"google..com.", "", errZeroSegLen},
		{"google.com.", "google.com.", nil},
		{".google.com.", "", errZeroSegLen},
		{"www..google.com.", "", errZeroSegLen},
		{"www.google.com.", "www.google.com.", nil},
	}

	for _, test := range tests {
		in := mustNewName(test.in)
		want := mustNewName(test.want)
		buf, err := in.pack(make([]byte, 0, 30), map[string]int{}, 0)
		if err != test.err {
			t.Errorf("Packing of %q: got err = %v, want err = %v", test.in, err, test.err)
			continue
		}
		if test.err != nil {
			continue
		}
		var got Name
		n, err := got.unpack(buf, 0)
		if err != nil {
			t.Errorf("Unpacking for %q failed: %v", test.in, err)
			continue
		}
		if n != len(buf) {
			t.Errorf(
				"Unpacked different amount than packed for %q: got n = %d, want = %d",
				test.in,
				n,
				len(buf),
			)
		}
		if got != want {
			t.Errorf("Unpacking packing of %q: got = %#v, want = %#v", test.in, got, want)
		}
	}
}

func TestIncompressibleName(t *testing.T) {
	name := mustNewName("example.com.")
	compression := map[string]int{}
	buf, err := name.pack(make([]byte, 0, 100), compression, 0)
	if err != nil {
		t.Fatal("First packing failed:", err)
	}
	buf, err = name.pack(buf, compression, 0)
	if err != nil {
		t.Fatal("Second packing failed:", err)
	}
	var n1 Name
	off, err := n1.unpackCompressed(buf, 0, false /* allowCompression */)
	if err != nil {
		t.Fatal("Unpacking incompressible name without pointers failed:", err)
	}
	var n2 Name
	if _, err := n2.unpackCompressed(buf, off, false /* allowCompression */); err != errCompressedSRV {
		t.Errorf("Unpacking compressed incompressible name with pointers: got err = %v, want = %v", err, errCompressedSRV)
	}
}

func checkErrorPrefix(err error, prefix string) bool {
	e, ok := err.(*nestedError)
	return ok && e.s == prefix
}

func TestHeaderUnpackError(t *testing.T) {
	wants := []string{
		"id",
		"bits",
		"questions",
		"answers",
		"authorities",
		"additionals",
	}
	var buf []byte
	var h header
	for _, want := range wants {
		n, err := h.unpack(buf, 0)
		if n != 0 || !checkErrorPrefix(err, want) {
			t.Errorf("got h.unpack([%d]byte, 0) = %d, %v, want = 0, %s", len(buf), n, err, want)
		}
		buf = append(buf, 0, 0)
	}
}

func TestParserStart(t *testing.T) {
	const want = "unpacking header"
	var p Parser
	for i := 0; i <= 1; i++ {
		_, err := p.Start([]byte{})
		if !checkErrorPrefix(err, want) {
			t.Errorf("got p.Start(nil) = _, %v, want = _, %s", err, want)
		}
	}
}

func TestResourceNotStarted(t *testing.T) {
	tests := []struct {
		name string
		fn   func(*Parser) error
	}{
		{"CNAMEResource", func(p *Parser) error { _, err := p.CNAMEResource(); return err }},
		{"MXResource", func(p *Parser) error { _, err := p.MXResource(); return err }},
		{"NSResource", func(p *Parser) error { _, err := p.NSResource(); return err }},
		{"PTRResource", func(p *Parser) error { _, err := p.PTRResource(); return err }},
		{"SOAResource", func(p *Parser) error { _, err := p.SOAResource(); return err }},
		{"TXTResource", func(p *Parser) error { _, err := p.TXTResource(); return err }},
		{"SRVResource", func(p *Parser) error { _, err := p.SRVResource(); return err }},
		{"AResource", func(p *Parser) error { _, err := p.AResource(); return err }},
		{"AAAAResource", func(p *Parser) error { _, err := p.AAAAResource(); return err }},
	}

	for _, test := range tests {
		if err := test.fn(&Parser{}); err != ErrNotStarted {
			t.Errorf("got _, %v = p.%s(), want = _, %v", err, test.name, ErrNotStarted)
		}
	}
}

func TestDNSPackUnpack(t *testing.T) {
	wants := []Message{
		{
			Questions: []Question{
				{
					Name:  mustNewName("."),
					Type:  TypeAAAA,
					Class: ClassINET,
				},
			},
			Answers:     []Resource{},
			Authorities: []Resource{},
			Additionals: []Resource{},
		},
		largeTestMsg(),
	}
	for i, want := range wants {
		b, err := want.Pack()
		if err != nil {
			t.Fatalf("%d: packing failed: %v", i, err)
		}
		var got Message
		err = got.Unpack(b)
		if err != nil {
			t.Fatalf("%d: unpacking failed: %v", i, err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%d: got = %+v, want = %+v", i, &got, &want)
		}
	}
}

func TestDNSAppendPackUnpack(t *testing.T) {
	wants := []Message{
		{
			Questions: []Question{
				{
					Name:  mustNewName("."),
					Type:  TypeAAAA,
					Class: ClassINET,
				},
			},
			Answers:     []Resource{},
			Authorities: []Resource{},
			Additionals: []Resource{},
		},
		largeTestMsg(),
	}
	for i, want := range wants {
		b := make([]byte, 2, 514)
		b, err := want.AppendPack(b)
		if err != nil {
			t.Fatalf("%d: packing failed: %v", i, err)
		}
		b = b[2:]
		var got Message
		err = got.Unpack(b)
		if err != nil {
			t.Fatalf("%d: unpacking failed: %v", i, err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%d: got = %+v, want = %+v", i, &got, &want)
		}
	}
}

func TestSkipAll(t *testing.T) {
	msg := largeTestMsg()
	buf, err := msg.Pack()
	if err != nil {
		t.Fatal("Packing large test message:", err)
	}
	var p Parser
	if _, err := p.Start(buf); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		f    func() error
	}{
		{"SkipAllQuestions", p.SkipAllQuestions},
		{"SkipAllAnswers", p.SkipAllAnswers},
		{"SkipAllAuthorities", p.SkipAllAuthorities},
		{"SkipAllAdditionals", p.SkipAllAdditionals},
	}
	for _, test := range tests {
		for i := 1; i <= 3; i++ {
			if err := test.f(); err != nil {
				t.Errorf("Call #%d to %s(): %v", i, test.name, err)
			}
		}
	}
}

func TestSkipEach(t *testing.T) {
	msg := smallTestMsg()

	buf, err := msg.Pack()
	if err != nil {
		t.Fatal("Packing test message:", err)
	}
	var p Parser
	if _, err := p.Start(buf); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		f    func() error
	}{
		{"SkipQuestion", p.SkipQuestion},
		{"SkipAnswer", p.SkipAnswer},
		{"SkipAuthority", p.SkipAuthority},
		{"SkipAdditional", p.SkipAdditional},
	}
	for _, test := range tests {
		if err := test.f(); err != nil {
			t.Errorf("First call: got %s() = %v, want = %v", test.name, err, nil)
		}
		if err := test.f(); err != ErrSectionDone {
			t.Errorf("Second call: got %s() = %v, want = %v", test.name, err, ErrSectionDone)
		}
	}
}

func TestSkipAfterRead(t *testing.T) {
	msg := smallTestMsg()

	buf, err := msg.Pack()
	if err != nil {
		t.Fatal("Packing test message:", err)
	}
	var p Parser
	if _, err := p.Start(buf); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		skip func() error
		read func() error
	}{
		{"Question", p.SkipQuestion, func() error { _, err := p.Question(); return err }},
		{"Answer", p.SkipAnswer, func() error { _, err := p.Answer(); return err }},
		{"Authority", p.SkipAuthority, func() error { _, err := p.Authority(); return err }},
		{"Additional", p.SkipAdditional, func() error { _, err := p.Additional(); return err }},
	}
	for _, test := range tests {
		if err := test.read(); err != nil {
			t.Errorf("Got %s() = _, %v, want = _, %v", test.name, err, nil)
		}
		if err := test.skip(); err != ErrSectionDone {
			t.Errorf("Got Skip%s() = %v, want = %v", test.name, err, ErrSectionDone)
		}
	}
}

func TestSkipNotStarted(t *testing.T) {
	var p Parser

	tests := []struct {
		name string
		f    func() error
	}{
		{"SkipAllQuestions", p.SkipAllQuestions},
		{"SkipAllAnswers", p.SkipAllAnswers},
		{"SkipAllAuthorities", p.SkipAllAuthorities},
		{"SkipAllAdditionals", p.SkipAllAdditionals},
	}
	for _, test := range tests {
		if err := test.f(); err != ErrNotStarted {
			t.Errorf("Got %s() = %v, want = %v", test.name, err, ErrNotStarted)
		}
	}
}

func TestTooManyRecords(t *testing.T) {
	const recs = int(^uint16(0)) + 1
	tests := []struct {
		name string
		msg  Message
		want error
	}{
		{
			"Questions",
			Message{
				Questions: make([]Question, recs),
			},
			errTooManyQuestions,
		},
		{
			"Answers",
			Message{
				Answers: make([]Resource, recs),
			},
			errTooManyAnswers,
		},
		{
			"Authorities",
			Message{
				Authorities: make([]Resource, recs),
			},
			errTooManyAuthorities,
		},
		{
			"Additionals",
			Message{
				Additionals: make([]Resource, recs),
			},
			errTooManyAdditionals,
		},
	}

	for _, test := range tests {
		if _, got := test.msg.Pack(); got != test.want {
			t.Errorf("Packing %d %s: got = %v, want = %v", recs, test.name, got, test.want)
		}
	}
}

func TestVeryLongTxt(t *testing.T) {
	want := Resource{
		ResourceHeader{
			Name:  mustNewName("foo.bar.example.com."),
			Type:  TypeTXT,
			Class: ClassINET,
		},
		&TXTResource{[]string{
			"",
			"",
			"foo bar",
			"",
			"www.example.com",
			"www.example.com.",
			strings.Repeat(".", 255),
		}},
	}
	buf, err := want.pack(make([]byte, 0, 8000), map[string]int{}, 0)
	if err != nil {
		t.Fatal("Packing failed:", err)
	}
	var got Resource
	off, err := got.Header.unpack(buf, 0)
	if err != nil {
		t.Fatal("Unpacking ResourceHeader failed:", err)
	}
	body, n, err := unpackResourceBody(buf, off, got.Header)
	if err != nil {
		t.Fatal("Unpacking failed:", err)
	}
	got.Body = body
	if n != len(buf) {
		t.Errorf("Unpacked different amount than packed: got n = %d, want = %d", n, len(buf))
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Got = %#v, want = %#v", got, want)
	}
}

func TestTooLongTxt(t *testing.T) {
	rb := TXTResource{[]string{strings.Repeat(".", 256)}}
	if _, err := rb.pack(make([]byte, 0, 8000), map[string]int{}, 0); err != errStringTooLong {
		t.Errorf("Packing TXTRecord with 256 character string: got err = %v, want = %v", err, errStringTooLong)
	}
}

func TestStartAppends(t *testing.T) {
	buf := make([]byte, 2, 514)
	wantBuf := []byte{4, 44}
	copy(buf, wantBuf)

	b := NewBuilder(buf, Header{})
	b.EnableCompression()

	buf, err := b.Finish()
	if err != nil {
		t.Fatal("Building failed:", err)
	}
	if got, want := len(buf), headerLen+2; got != want {
		t.Errorf("Got len(buf} = %d, want = %d", got, want)
	}
	if string(buf[:2]) != string(wantBuf) {
		t.Errorf("Original data not preserved, got = %v, want = %v", buf[:2], wantBuf)
	}
}

func TestStartError(t *testing.T) {
	tests := []struct {
		name string
		fn   func(*Builder) error
	}{
		{"Questions", func(b *Builder) error { return b.StartQuestions() }},
		{"Answers", func(b *Builder) error { return b.StartAnswers() }},
		{"Authorities", func(b *Builder) error { return b.StartAuthorities() }},
		{"Additionals", func(b *Builder) error { return b.StartAdditionals() }},
	}

	envs := []struct {
		name string
		fn   func() *Builder
		want error
	}{
		{"sectionNotStarted", func() *Builder { return &Builder{section: sectionNotStarted} }, ErrNotStarted},
		{"sectionDone", func() *Builder { return &Builder{section: sectionDone} }, ErrSectionDone},
	}

	for _, env := range envs {
		for _, test := range tests {
			if got := test.fn(env.fn()); got != env.want {
				t.Errorf("got Builder{%s}.Start%s = %v, want = %v", env.name, test.name, got, env.want)
			}
		}
	}
}

func TestBuilderResourceError(t *testing.T) {
	tests := []struct {
		name string
		fn   func(*Builder) error
	}{
		{"CNAMEResource", func(b *Builder) error { return b.CNAMEResource(ResourceHeader{}, CNAMEResource{}) }},
		{"MXResource", func(b *Builder) error { return b.MXResource(ResourceHeader{}, MXResource{}) }},
		{"NSResource", func(b *Builder) error { return b.NSResource(ResourceHeader{}, NSResource{}) }},
		{"PTRResource", func(b *Builder) error { return b.PTRResource(ResourceHeader{}, PTRResource{}) }},
		{"SOAResource", func(b *Builder) error { return b.SOAResource(ResourceHeader{}, SOAResource{}) }},
		{"TXTResource", func(b *Builder) error { return b.TXTResource(ResourceHeader{}, TXTResource{}) }},
		{"SRVResource", func(b *Builder) error { return b.SRVResource(ResourceHeader{}, SRVResource{}) }},
		{"AResource", func(b *Builder) error { return b.AResource(ResourceHeader{}, AResource{}) }},
		{"AAAAResource", func(b *Builder) error { return b.AAAAResource(ResourceHeader{}, AAAAResource{}) }},
	}

	envs := []struct {
		name string
		fn   func() *Builder
		want error
	}{
		{"sectionNotStarted", func() *Builder { return &Builder{section: sectionNotStarted} }, ErrNotStarted},
		{"sectionHeader", func() *Builder { return &Builder{section: sectionHeader} }, ErrNotStarted},
		{"sectionQuestions", func() *Builder { return &Builder{section: sectionQuestions} }, ErrNotStarted},
		{"sectionDone", func() *Builder { return &Builder{section: sectionDone} }, ErrSectionDone},
	}

	for _, env := range envs {
		for _, test := range tests {
			if got := test.fn(env.fn()); got != env.want {
				t.Errorf("got Builder{%s}.%s = %v, want = %v", env.name, test.name, got, env.want)
			}
		}
	}
}

func TestFinishError(t *testing.T) {
	var b Builder
	want := ErrNotStarted
	if _, got := b.Finish(); got != want {
		t.Errorf("got Builder{}.Finish() = %v, want = %v", got, want)
	}
}

func TestBuilder(t *testing.T) {
	msg := largeTestMsg()
	want, err := msg.Pack()
	if err != nil {
		t.Fatal("Packing without builder:", err)
	}

	b := NewBuilder(nil, msg.Header)
	b.EnableCompression()

	if err := b.StartQuestions(); err != nil {
		t.Fatal("b.StartQuestions():", err)
	}
	for _, q := range msg.Questions {
		if err := b.Question(q); err != nil {
			t.Fatalf("b.Question(%#v): %v", q, err)
		}
	}

	if err := b.StartAnswers(); err != nil {
		t.Fatal("b.StartAnswers():", err)
	}
	for _, a := range msg.Answers {
		switch a.Header.Type {
		case TypeA:
			if err := b.AResource(a.Header, *a.Body.(*AResource)); err != nil {
				t.Fatalf("b.AResource(%#v): %v", a, err)
			}
		case TypeNS:
			if err := b.NSResource(a.Header, *a.Body.(*NSResource)); err != nil {
				t.Fatalf("b.NSResource(%#v): %v", a, err)
			}
		case TypeCNAME:
			if err := b.CNAMEResource(a.Header, *a.Body.(*CNAMEResource)); err != nil {
				t.Fatalf("b.CNAMEResource(%#v): %v", a, err)
			}
		case TypeSOA:
			if err := b.SOAResource(a.Header, *a.Body.(*SOAResource)); err != nil {
				t.Fatalf("b.SOAResource(%#v): %v", a, err)
			}
		case TypePTR:
			if err := b.PTRResource(a.Header, *a.Body.(*PTRResource)); err != nil {
				t.Fatalf("b.PTRResource(%#v): %v", a, err)
			}
		case TypeMX:
			if err := b.MXResource(a.Header, *a.Body.(*MXResource)); err != nil {
				t.Fatalf("b.MXResource(%#v): %v", a, err)
			}
		case TypeTXT:
			if err := b.TXTResource(a.Header, *a.Body.(*TXTResource)); err != nil {
				t.Fatalf("b.TXTResource(%#v): %v", a, err)
			}
		case TypeAAAA:
			if err := b.AAAAResource(a.Header, *a.Body.(*AAAAResource)); err != nil {
				t.Fatalf("b.AAAAResource(%#v): %v", a, err)
			}
		case TypeSRV:
			if err := b.SRVResource(a.Header, *a.Body.(*SRVResource)); err != nil {
				t.Fatalf("b.SRVResource(%#v): %v", a, err)
			}
		}
	}

	if err := b.StartAuthorities(); err != nil {
		t.Fatal("b.StartAuthorities():", err)
	}
	for _, a := range msg.Authorities {
		if err := b.NSResource(a.Header, *a.Body.(*NSResource)); err != nil {
			t.Fatalf("b.NSResource(%#v): %v", a, err)
		}
	}

	if err := b.StartAdditionals(); err != nil {
		t.Fatal("b.StartAdditionals():", err)
	}
	for _, a := range msg.Additionals {
		if err := b.TXTResource(a.Header, *a.Body.(*TXTResource)); err != nil {
			t.Fatalf("b.TXTResource(%#v): %v", a, err)
		}
	}

	got, err := b.Finish()
	if err != nil {
		t.Fatal("b.Finish():", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("Got from Builder: %#v\nwant = %#v", got, want)
	}
}

func TestResourcePack(t *testing.T) {
	for _, tt := range []struct {
		m   Message
		err error
	}{
		{
			Message{
				Questions: []Question{
					{
						Name:  mustNewName("."),
						Type:  TypeAAAA,
						Class: ClassINET,
					},
				},
				Answers: []Resource{{ResourceHeader{}, nil}},
			},
			&nestedError{"packing Answer", errNilResouceBody},
		},
		{
			Message{
				Questions: []Question{
					{
						Name:  mustNewName("."),
						Type:  TypeAAAA,
						Class: ClassINET,
					},
				},
				Authorities: []Resource{{ResourceHeader{}, (*NSResource)(nil)}},
			},
			&nestedError{"packing Authority",
				&nestedError{"ResourceHeader",
					&nestedError{"Name", errNonCanonicalName},
				},
			},
		},
		{
			Message{
				Questions: []Question{
					{
						Name:  mustNewName("."),
						Type:  TypeA,
						Class: ClassINET,
					},
				},
				Additionals: []Resource{{ResourceHeader{}, nil}},
			},
			&nestedError{"packing Additional", errNilResouceBody},
		},
	} {
		_, err := tt.m.Pack()
		if !reflect.DeepEqual(err, tt.err) {
			t.Errorf("got %v for %v; want %v", err, tt.m, tt.err)
		}
	}
}

func benchmarkParsingSetup() ([]byte, error) {
	name := mustNewName("foo.bar.example.com.")
	msg := Message{
		Header: Header{Response: true, Authoritative: true},
		Questions: []Question{
			{
				Name:  name,
				Type:  TypeA,
				Class: ClassINET,
			},
		},
		Answers: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Class: ClassINET,
				},
				&AResource{[4]byte{}},
			},
			{
				ResourceHeader{
					Name:  name,
					Class: ClassINET,
				},
				&AAAAResource{[16]byte{}},
			},
			{
				ResourceHeader{
					Name:  name,
					Class: ClassINET,
				},
				&CNAMEResource{name},
			},
			{
				ResourceHeader{
					Name:  name,
					Class: ClassINET,
				},
				&NSResource{name},
			},
		},
	}

	buf, err := msg.Pack()
	if err != nil {
		return nil, fmt.Errorf("msg.Pack(): %v", err)
	}
	return buf, nil
}

func benchmarkParsing(tb testing.TB, buf []byte) {
	var p Parser
	if _, err := p.Start(buf); err != nil {
		tb.Fatal("p.Start(buf):", err)
	}

	for {
		_, err := p.Question()
		if err == ErrSectionDone {
			break
		}
		if err != nil {
			tb.Fatal("p.Question():", err)
		}
	}

	for {
		h, err := p.AnswerHeader()
		if err == ErrSectionDone {
			break
		}
		if err != nil {
			panic(err)
		}

		switch h.Type {
		case TypeA:
			if _, err := p.AResource(); err != nil {
				tb.Fatal("p.AResource():", err)
			}
		case TypeAAAA:
			if _, err := p.AAAAResource(); err != nil {
				tb.Fatal("p.AAAAResource():", err)
			}
		case TypeCNAME:
			if _, err := p.CNAMEResource(); err != nil {
				tb.Fatal("p.CNAMEResource():", err)
			}
		case TypeNS:
			if _, err := p.NSResource(); err != nil {
				tb.Fatal("p.NSResource():", err)
			}
		default:
			tb.Fatalf("unknown type: %T", h)
		}
	}
}

func BenchmarkParsing(b *testing.B) {
	buf, err := benchmarkParsingSetup()
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchmarkParsing(b, buf)
	}
}

func TestParsingAllocs(t *testing.T) {
	buf, err := benchmarkParsingSetup()
	if err != nil {
		t.Fatal(err)
	}

	if allocs := testing.AllocsPerRun(100, func() { benchmarkParsing(t, buf) }); allocs > 0.5 {
		t.Errorf("Allocations during parsing: got = %f, want ~0", allocs)
	}
}

func benchmarkBuildingSetup() (Name, []byte) {
	name := mustNewName("foo.bar.example.com.")
	buf := make([]byte, 0, packStartingCap)
	return name, buf
}

func benchmarkBuilding(tb testing.TB, name Name, buf []byte) {
	bld := NewBuilder(buf, Header{Response: true, Authoritative: true})

	if err := bld.StartQuestions(); err != nil {
		tb.Fatal("bld.StartQuestions():", err)
	}
	q := Question{
		Name:  name,
		Type:  TypeA,
		Class: ClassINET,
	}
	if err := bld.Question(q); err != nil {
		tb.Fatalf("bld.Question(%+v): %v", q, err)
	}

	hdr := ResourceHeader{
		Name:  name,
		Class: ClassINET,
	}
	if err := bld.StartAnswers(); err != nil {
		tb.Fatal("bld.StartQuestions():", err)
	}

	ar := AResource{[4]byte{}}
	if err := bld.AResource(hdr, ar); err != nil {
		tb.Fatalf("bld.AResource(%+v, %+v): %v", hdr, ar, err)
	}

	aaar := AAAAResource{[16]byte{}}
	if err := bld.AAAAResource(hdr, aaar); err != nil {
		tb.Fatalf("bld.AAAAResource(%+v, %+v): %v", hdr, aaar, err)
	}

	cnr := CNAMEResource{name}
	if err := bld.CNAMEResource(hdr, cnr); err != nil {
		tb.Fatalf("bld.CNAMEResource(%+v, %+v): %v", hdr, cnr, err)
	}

	nsr := NSResource{name}
	if err := bld.NSResource(hdr, nsr); err != nil {
		tb.Fatalf("bld.NSResource(%+v, %+v): %v", hdr, nsr, err)
	}

	if _, err := bld.Finish(); err != nil {
		tb.Fatal("bld.Finish():", err)
	}
}

func BenchmarkBuilding(b *testing.B) {
	name, buf := benchmarkBuildingSetup()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchmarkBuilding(b, name, buf)
	}
}

func TestBuildingAllocs(t *testing.T) {
	name, buf := benchmarkBuildingSetup()
	if allocs := testing.AllocsPerRun(100, func() { benchmarkBuilding(t, name, buf) }); allocs > 0.5 {
		t.Errorf("Allocations during building: got = %f, want ~0", allocs)
	}
}

func smallTestMsg() Message {
	name := mustNewName("example.com.")
	return Message{
		Header: Header{Response: true, Authoritative: true},
		Questions: []Question{
			{
				Name:  name,
				Type:  TypeA,
				Class: ClassINET,
			},
		},
		Answers: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeA,
					Class: ClassINET,
				},
				&AResource{[4]byte{127, 0, 0, 1}},
			},
		},
		Authorities: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeA,
					Class: ClassINET,
				},
				&AResource{[4]byte{127, 0, 0, 1}},
			},
		},
		Additionals: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeA,
					Class: ClassINET,
				},
				&AResource{[4]byte{127, 0, 0, 1}},
			},
		},
	}
}

func BenchmarkPack(b *testing.B) {
	msg := largeTestMsg()

	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if _, err := msg.Pack(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAppendPack(b *testing.B) {
	msg := largeTestMsg()
	buf := make([]byte, 0, packStartingCap)

	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if _, err := msg.AppendPack(buf[:0]); err != nil {
			b.Fatal(err)
		}
	}
}

func largeTestMsg() Message {
	name := mustNewName("foo.bar.example.com.")
	return Message{
		Header: Header{Response: true, Authoritative: true},
		Questions: []Question{
			{
				Name:  name,
				Type:  TypeA,
				Class: ClassINET,
			},
		},
		Answers: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeA,
					Class: ClassINET,
				},
				&AResource{[4]byte{127, 0, 0, 1}},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeA,
					Class: ClassINET,
				},
				&AResource{[4]byte{127, 0, 0, 2}},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeAAAA,
					Class: ClassINET,
				},
				&AAAAResource{[16]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeCNAME,
					Class: ClassINET,
				},
				&CNAMEResource{mustNewName("alias.example.com.")},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeSOA,
					Class: ClassINET,
				},
				&SOAResource{
					NS:      mustNewName("ns1.example.com."),
					MBox:    mustNewName("mb.example.com."),
					Serial:  1,
					Refresh: 2,
					Retry:   3,
					Expire:  4,
					MinTTL:  5,
				},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypePTR,
					Class: ClassINET,
				},
				&PTRResource{mustNewName("ptr.example.com.")},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeMX,
					Class: ClassINET,
				},
				&MXResource{
					7,
					mustNewName("mx.example.com."),
				},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeSRV,
					Class: ClassINET,
				},
				&SRVResource{
					8,
					9,
					11,
					mustNewName("srv.example.com."),
				},
			},
		},
		Authorities: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeNS,
					Class: ClassINET,
				},
				&NSResource{mustNewName("ns1.example.com.")},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeNS,
					Class: ClassINET,
				},
				&NSResource{mustNewName("ns2.example.com.")},
			},
		},
		Additionals: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeTXT,
					Class: ClassINET,
				},
				&TXTResource{[]string{"So Long, and Thanks for All the Fish"}},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeTXT,
					Class: ClassINET,
				},
				&TXTResource{[]string{"Hamster Huey and the Gooey Kablooie"}},
			},
		},
	}
}
