// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"container/vector"
	"fmt"
	"io"
	"io/ioutil"
	"json"
	"os"
	"strings"
	"testing"
)

type Test struct {
	in, out, err string
}

type T struct {
	Item  string
	Value string
}

type U struct {
	Mp map[string]int
}

type S struct {
	Header        string
	Integer       int
	Raw           string
	InnerT        T
	InnerPointerT *T
	Data          []T
	Pdata         []*T
	Empty         []*T
	Emptystring   string
	Null          []*T
	Vec           *vector.Vector
	True          bool
	False         bool
	Mp            map[string]string
	JSON          interface{}
	Innermap      U
	Stringmap     map[string]string
	Bytes         []byte
	Iface         interface{}
	Ifaceptr      interface{}
}

func (s *S) PointerMethod() string { return "ptrmethod!" }

func (s S) ValueMethod() string { return "valmethod!" }

var t1 = T{"ItemNumber1", "ValueNumber1"}
var t2 = T{"ItemNumber2", "ValueNumber2"}

func uppercase(v interface{}) string {
	s := v.(string)
	t := ""
	for i := 0; i < len(s); i++ {
		c := s[i]
		if 'a' <= c && c <= 'z' {
			c = c + 'A' - 'a'
		}
		t += string(c)
	}
	return t
}

func plus1(v interface{}) string {
	i := v.(int)
	return fmt.Sprint(i + 1)
}

func writer(f func(interface{}) string) func(io.Writer, string, ...interface{}) {
	return func(w io.Writer, format string, v ...interface{}) {
		if len(v) != 1 {
			panic("test writer expected one arg")
		}
		io.WriteString(w, f(v[0]))
	}
}

func multiword(w io.Writer, format string, value ...interface{}) {
	for _, v := range value {
		fmt.Fprintf(w, "<%v>", v)
	}
}

var formatters = FormatterMap{
	"uppercase": writer(uppercase),
	"+1":        writer(plus1),
	"multiword": multiword,
}

var tests = []*Test{
	// Simple
	&Test{"", "", ""},
	&Test{"abc", "abc", ""},
	&Test{"abc\ndef\n", "abc\ndef\n", ""},
	&Test{" {.meta-left}   \n", "{", ""},
	&Test{" {.meta-right}   \n", "}", ""},
	&Test{" {.space}   \n", " ", ""},
	&Test{" {.tab}   \n", "\t", ""},
	&Test{"     {#comment}   \n", "", ""},
	&Test{"\tSome Text\t\n", "\tSome Text\t\n", ""},
	&Test{" {.meta-right} {.meta-right} {.meta-right} \n", " } } } \n", ""},

	// Variables at top level
	&Test{
		in: "{Header}={Integer}\n",

		out: "Header=77\n",
	},

	// Method at top level
	&Test{
		in: "ptrmethod={PointerMethod}\n",

		out: "ptrmethod=ptrmethod!\n",
	},

	&Test{
		in: "valmethod={ValueMethod}\n",

		out: "valmethod=valmethod!\n",
	},

	// Section
	&Test{
		in: "{.section Data }\n" +
			"some text for the section\n" +
			"{.end}\n",

		out: "some text for the section\n",
	},
	&Test{
		in: "{.section Data }\n" +
			"{Header}={Integer}\n" +
			"{.end}\n",

		out: "Header=77\n",
	},
	&Test{
		in: "{.section Pdata }\n" +
			"{Header}={Integer}\n" +
			"{.end}\n",

		out: "Header=77\n",
	},
	&Test{
		in: "{.section Pdata }\n" +
			"data present\n" +
			"{.or}\n" +
			"data not present\n" +
			"{.end}\n",

		out: "data present\n",
	},
	&Test{
		in: "{.section Empty }\n" +
			"data present\n" +
			"{.or}\n" +
			"data not present\n" +
			"{.end}\n",

		out: "data not present\n",
	},
	&Test{
		in: "{.section Null }\n" +
			"data present\n" +
			"{.or}\n" +
			"data not present\n" +
			"{.end}\n",

		out: "data not present\n",
	},
	&Test{
		in: "{.section Pdata }\n" +
			"{Header}={Integer}\n" +
			"{.section @ }\n" +
			"{Header}={Integer}\n" +
			"{.end}\n" +
			"{.end}\n",

		out: "Header=77\n" +
			"Header=77\n",
	},

	&Test{
		in: "{.section Data}{.end} {Header}\n",

		out: " Header\n",
	},

	&Test{
		in: "{.section Integer}{@}{.end}",

		out: "77",
	},


	// Repeated
	&Test{
		in: "{.section Pdata }\n" +
			"{.repeated section @ }\n" +
			"{Item}={Value}\n" +
			"{.end}\n" +
			"{.end}\n",

		out: "ItemNumber1=ValueNumber1\n" +
			"ItemNumber2=ValueNumber2\n",
	},
	&Test{
		in: "{.section Pdata }\n" +
			"{.repeated section @ }\n" +
			"{Item}={Value}\n" +
			"{.or}\n" +
			"this should not appear\n" +
			"{.end}\n" +
			"{.end}\n",

		out: "ItemNumber1=ValueNumber1\n" +
			"ItemNumber2=ValueNumber2\n",
	},
	&Test{
		in: "{.section @ }\n" +
			"{.repeated section Empty }\n" +
			"{Item}={Value}\n" +
			"{.or}\n" +
			"this should appear: empty field\n" +
			"{.end}\n" +
			"{.end}\n",

		out: "this should appear: empty field\n",
	},
	&Test{
		in: "{.repeated section Pdata }\n" +
			"{Item}\n" +
			"{.alternates with}\n" +
			"is\nover\nmultiple\nlines\n" +
			"{.end}\n",

		out: "ItemNumber1\n" +
			"is\nover\nmultiple\nlines\n" +
			"ItemNumber2\n",
	},
	&Test{
		in: "{.repeated section Pdata }\n" +
			"{Item}\n" +
			"{.alternates with}\n" +
			"is\nover\nmultiple\nlines\n" +
			" {.end}\n",

		out: "ItemNumber1\n" +
			"is\nover\nmultiple\nlines\n" +
			"ItemNumber2\n",
	},
	&Test{
		in: "{.section Pdata }\n" +
			"{.repeated section @ }\n" +
			"{Item}={Value}\n" +
			"{.alternates with}DIVIDER\n" +
			"{.or}\n" +
			"this should not appear\n" +
			"{.end}\n" +
			"{.end}\n",

		out: "ItemNumber1=ValueNumber1\n" +
			"DIVIDER\n" +
			"ItemNumber2=ValueNumber2\n",
	},
	&Test{
		in: "{.repeated section Vec }\n" +
			"{@}\n" +
			"{.end}\n",

		out: "elt1\n" +
			"elt2\n",
	},
	// Same but with a space before {.end}: was a bug.
	&Test{
		in: "{.repeated section Vec }\n" +
			"{@} {.end}\n",

		out: "elt1 elt2 \n",
	},
	&Test{
		in: "{.repeated section Integer}{.end}",

		err: "line 1: .repeated: cannot repeat Integer (type int)",
	},

	// Nested names
	&Test{
		in: "{.section @ }\n" +
			"{InnerT.Item}={InnerT.Value}\n" +
			"{.end}",

		out: "ItemNumber1=ValueNumber1\n",
	},
	&Test{
		in: "{.section @ }\n" +
			"{InnerT.Item}={.section InnerT}{.section Value}{@}{.end}{.end}\n" +
			"{.end}",

		out: "ItemNumber1=ValueNumber1\n",
	},


	// Formatters
	&Test{
		in: "{.section Pdata }\n" +
			"{Header|uppercase}={Integer|+1}\n" +
			"{Header|html}={Integer|str}\n" +
			"{.end}\n",

		out: "HEADER=78\n" +
			"Header=77\n",
	},

	&Test{
		in: "{.section Pdata }\n" +
			"{Header|uppercase}={Integer Header|multiword}\n" +
			"{Header|html}={Header Integer|multiword}\n" +
			"{Header|html}={Header Integer}\n" +
			"{.end}\n",

		out: "HEADER=<77><Header>\n" +
			"Header=<Header><77>\n" +
			"Header=Header77\n",
	},

	&Test{
		in: "{Raw}\n" +
			"{Raw|html}\n",

		out: "&<>!@ #$%^\n" +
			"&amp;&lt;&gt;!@ #$%^\n",
	},

	&Test{
		in: "{.section Emptystring}emptystring{.end}\n" +
			"{.section Header}header{.end}\n",

		out: "\nheader\n",
	},

	&Test{
		in: "{.section True}1{.or}2{.end}\n" +
			"{.section False}3{.or}4{.end}\n",

		out: "1\n4\n",
	},

	&Test{
		in: "{Bytes}",

		out: "hello",
	},

	// Maps

	&Test{
		in: "{Mp.mapkey}\n",

		out: "Ahoy!\n",
	},
	&Test{
		in: "{Innermap.Mp.innerkey}\n",

		out: "55\n",
	},
	&Test{
		in: "{.section Innermap}{.section Mp}{innerkey}{.end}{.end}\n",

		out: "55\n",
	},
	&Test{
		in: "{.section JSON}{.repeated section maps}{a}{b}{.end}{.end}\n",

		out: "1234\n",
	},
	&Test{
		in: "{Stringmap.stringkey1}\n",

		out: "stringresult\n",
	},
	&Test{
		in: "{.repeated section Stringmap}\n" +
			"{@}\n" +
			"{.end}",

		out: "stringresult\n" +
			"stringresult\n",
	},
	&Test{
		in: "{.repeated section Stringmap}\n" +
			"\t{@}\n" +
			"{.end}",

		out: "\tstringresult\n" +
			"\tstringresult\n",
	},

	// Interface values

	&Test{
		in: "{Iface}",

		out: "[1 2 3]",
	},
	&Test{
		in: "{.repeated section Iface}{@}{.alternates with} {.end}",

		out: "1 2 3",
	},
	&Test{
		in: "{.section Iface}{@}{.end}",

		out: "[1 2 3]",
	},
	&Test{
		in: "{.section Ifaceptr}{Item} {Value}{.end}",

		out: "Item Value",
	},
}

func TestAll(t *testing.T) {
	// Parse
	testAll(t, func(test *Test) (*Template, os.Error) { return Parse(test.in, formatters) })
	// ParseFile
	testAll(t, func(test *Test) (*Template, os.Error) {
		err := ioutil.WriteFile("_test/test.tmpl", []byte(test.in), 0600)
		if err != nil {
			t.Error("unexpected write error:", err)
			return nil, err
		}
		return ParseFile("_test/test.tmpl", formatters)
	})
	// tmpl.ParseFile
	testAll(t, func(test *Test) (*Template, os.Error) {
		err := ioutil.WriteFile("_test/test.tmpl", []byte(test.in), 0600)
		if err != nil {
			t.Error("unexpected write error:", err)
			return nil, err
		}
		tmpl := New(formatters)
		return tmpl, tmpl.ParseFile("_test/test.tmpl")
	})
}

func testAll(t *testing.T, parseFunc func(*Test) (*Template, os.Error)) {
	s := new(S)
	// initialized by hand for clarity.
	s.Header = "Header"
	s.Integer = 77
	s.Raw = "&<>!@ #$%^"
	s.InnerT = t1
	s.Data = []T{t1, t2}
	s.Pdata = []*T{&t1, &t2}
	s.Empty = []*T{}
	s.Null = nil
	s.Vec = new(vector.Vector)
	s.Vec.Push("elt1")
	s.Vec.Push("elt2")
	s.True = true
	s.False = false
	s.Mp = make(map[string]string)
	s.Mp["mapkey"] = "Ahoy!"
	json.Unmarshal([]byte(`{"maps":[{"a":1,"b":2},{"a":3,"b":4}]}`), &s.JSON)
	s.Innermap.Mp = make(map[string]int)
	s.Innermap.Mp["innerkey"] = 55
	s.Stringmap = make(map[string]string)
	s.Stringmap["stringkey1"] = "stringresult" // the same value so repeated section is order-independent
	s.Stringmap["stringkey2"] = "stringresult"
	s.Bytes = []byte("hello")
	s.Iface = []int{1, 2, 3}
	s.Ifaceptr = &T{"Item", "Value"}

	var buf bytes.Buffer
	for _, test := range tests {
		buf.Reset()
		tmpl, err := parseFunc(test)
		if err != nil {
			t.Error("unexpected parse error: ", err)
			continue
		}
		err = tmpl.Execute(s, &buf)
		if test.err == "" {
			if err != nil {
				t.Error("unexpected execute error:", err)
			}
		} else {
			if err == nil {
				t.Errorf("expected execute error %q, got nil", test.err)
			} else if err.String() != test.err {
				t.Errorf("expected execute error %q, got %q", test.err, err.String())
			}
		}
		if buf.String() != test.out {
			t.Errorf("for %q: expected %q got %q", test.in, test.out, buf.String())
		}
	}
}

func TestMapDriverType(t *testing.T) {
	mp := map[string]string{"footer": "Ahoy!"}
	tmpl, err := Parse("template: {footer}", nil)
	if err != nil {
		t.Error("unexpected parse error:", err)
	}
	var b bytes.Buffer
	err = tmpl.Execute(mp, &b)
	if err != nil {
		t.Error("unexpected execute error:", err)
	}
	s := b.String()
	expected := "template: Ahoy!"
	if s != expected {
		t.Errorf("failed passing string as data: expected %q got %q", "template: Ahoy!", s)
	}
}

func TestStringDriverType(t *testing.T) {
	tmpl, err := Parse("template: {@}", nil)
	if err != nil {
		t.Error("unexpected parse error:", err)
	}
	var b bytes.Buffer
	err = tmpl.Execute("hello", &b)
	if err != nil {
		t.Error("unexpected execute error:", err)
	}
	s := b.String()
	if s != "template: hello" {
		t.Errorf("failed passing string as data: expected %q got %q", "template: hello", s)
	}
}

func TestTwice(t *testing.T) {
	tmpl, err := Parse("template: {@}", nil)
	if err != nil {
		t.Error("unexpected parse error:", err)
	}
	var b bytes.Buffer
	err = tmpl.Execute("hello", &b)
	if err != nil {
		t.Error("unexpected parse error:", err)
	}
	s := b.String()
	text := "template: hello"
	if s != text {
		t.Errorf("failed passing string as data: expected %q got %q", text, s)
	}
	err = tmpl.Execute("hello", &b)
	if err != nil {
		t.Error("unexpected parse error:", err)
	}
	s = b.String()
	text += text
	if s != text {
		t.Errorf("failed passing string as data: expected %q got %q", text, s)
	}
}

func TestCustomDelims(t *testing.T) {
	// try various lengths.  zero should catch error.
	for i := 0; i < 7; i++ {
		for j := 0; j < 7; j++ {
			tmpl := New(nil)
			// first two chars deliberately the same to test equal left and right delims
			ldelim := "$!#$%^&"[0:i]
			rdelim := "$*&^%$!"[0:j]
			tmpl.SetDelims(ldelim, rdelim)
			// if braces, this would be template: {@}{.meta-left}{.meta-right}
			text := "template: " +
				ldelim + "@" + rdelim +
				ldelim + ".meta-left" + rdelim +
				ldelim + ".meta-right" + rdelim
			err := tmpl.Parse(text)
			if err != nil {
				if i == 0 || j == 0 { // expected
					continue
				}
				t.Error("unexpected parse error:", err)
			} else if i == 0 || j == 0 {
				t.Errorf("expected parse error for empty delimiter: %d %d %q %q", i, j, ldelim, rdelim)
				continue
			}
			var b bytes.Buffer
			err = tmpl.Execute("hello", &b)
			s := b.String()
			if s != "template: hello"+ldelim+rdelim {
				t.Errorf("failed delim check(%q %q) %q got %q", ldelim, rdelim, text, s)
			}
		}
	}
}

// Test that a variable evaluates to the field itself and does not further indirection
func TestVarIndirection(t *testing.T) {
	s := new(S)
	// initialized by hand for clarity.
	s.InnerPointerT = &t1

	var buf bytes.Buffer
	input := "{.section @}{InnerPointerT}{.end}"
	tmpl, err := Parse(input, nil)
	if err != nil {
		t.Fatal("unexpected parse error:", err)
	}
	err = tmpl.Execute(s, &buf)
	if err != nil {
		t.Fatal("unexpected execute error:", err)
	}
	expect := fmt.Sprintf("%v", &t1) // output should be hex address of t1
	if buf.String() != expect {
		t.Errorf("for %q: expected %q got %q", input, expect, buf.String())
	}
}

func TestHTMLFormatterWithByte(t *testing.T) {
	s := "Test string."
	b := []byte(s)
	var buf bytes.Buffer
	HTMLFormatter(&buf, "", b)
	bs := buf.String()
	if bs != s {
		t.Errorf("munged []byte, expected: %s got: %s", s, bs)
	}
}

type UF struct {
	I int
	s string
}

func TestReferenceToUnexported(t *testing.T) {
	u := &UF{3, "hello"}
	var buf bytes.Buffer
	input := "{.section @}{I}{s}{.end}"
	tmpl, err := Parse(input, nil)
	if err != nil {
		t.Fatal("unexpected parse error:", err)
	}
	err = tmpl.Execute(u, &buf)
	if err == nil {
		t.Fatal("expected execute error, got none")
	}
	if strings.Index(err.String(), "not exported") < 0 {
		t.Fatal("expected unexported error; got", err)
	}
}
