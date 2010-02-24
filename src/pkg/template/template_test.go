// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"container/vector"
	"fmt"
	"io"
	"strings"
	"testing"
)

type Test struct {
	in, out, err string
}

type T struct {
	item  string
	value string
}

type U struct {
	mp map[string]int
}

type S struct {
	header        string
	integer       int
	raw           string
	innerT        T
	innerPointerT *T
	data          []T
	pdata         []*T
	empty         []*T
	emptystring   string
	null          []*T
	vec           *vector.Vector
	true          bool
	false         bool
	mp            map[string]string
	innermap      U
	stringmap     map[string]string
	bytes         []byte
	iface         interface{}
}

func (s *S) pointerMethod() string { return "ptrmethod!" }

func (s S) valueMethod() string { return "valmethod!" }

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

func writer(f func(interface{}) string) func(io.Writer, interface{}, string) {
	return func(w io.Writer, v interface{}, format string) {
		io.WriteString(w, f(v))
	}
}


var formatters = FormatterMap{
	"uppercase": writer(uppercase),
	"+1": writer(plus1),
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

	// Variables at top level
	&Test{
		in: "{header}={integer}\n",

		out: "Header=77\n",
	},

	// Method at top level
	&Test{
		in: "ptrmethod={pointerMethod}\n",

		out: "ptrmethod=ptrmethod!\n",
	},

	&Test{
		in: "valmethod={valueMethod}\n",

		out: "valmethod=valmethod!\n",
	},

	// Section
	&Test{
		in: "{.section data }\n" +
			"some text for the section\n" +
			"{.end}\n",

		out: "some text for the section\n",
	},
	&Test{
		in: "{.section data }\n" +
			"{header}={integer}\n" +
			"{.end}\n",

		out: "Header=77\n",
	},
	&Test{
		in: "{.section pdata }\n" +
			"{header}={integer}\n" +
			"{.end}\n",

		out: "Header=77\n",
	},
	&Test{
		in: "{.section pdata }\n" +
			"data present\n" +
			"{.or}\n" +
			"data not present\n" +
			"{.end}\n",

		out: "data present\n",
	},
	&Test{
		in: "{.section empty }\n" +
			"data present\n" +
			"{.or}\n" +
			"data not present\n" +
			"{.end}\n",

		out: "data not present\n",
	},
	&Test{
		in: "{.section null }\n" +
			"data present\n" +
			"{.or}\n" +
			"data not present\n" +
			"{.end}\n",

		out: "data not present\n",
	},
	&Test{
		in: "{.section pdata }\n" +
			"{header}={integer}\n" +
			"{.section @ }\n" +
			"{header}={integer}\n" +
			"{.end}\n" +
			"{.end}\n",

		out: "Header=77\n" +
			"Header=77\n",
	},

	&Test{
		in: "{.section data}{.end} {header}\n",

		out: " Header\n",
	},

	// Repeated
	&Test{
		in: "{.section pdata }\n" +
			"{.repeated section @ }\n" +
			"{item}={value}\n" +
			"{.end}\n" +
			"{.end}\n",

		out: "ItemNumber1=ValueNumber1\n" +
			"ItemNumber2=ValueNumber2\n",
	},
	&Test{
		in: "{.section pdata }\n" +
			"{.repeated section @ }\n" +
			"{item}={value}\n" +
			"{.or}\n" +
			"this should not appear\n" +
			"{.end}\n" +
			"{.end}\n",

		out: "ItemNumber1=ValueNumber1\n" +
			"ItemNumber2=ValueNumber2\n",
	},
	&Test{
		in: "{.section @ }\n" +
			"{.repeated section empty }\n" +
			"{item}={value}\n" +
			"{.or}\n" +
			"this should appear: empty field\n" +
			"{.end}\n" +
			"{.end}\n",

		out: "this should appear: empty field\n",
	},
	&Test{
		in: "{.repeated section pdata }\n" +
			"{item}\n" +
			"{.alternates with}\n" +
			"is\nover\nmultiple\nlines\n" +
			"{.end}\n",

		out: "ItemNumber1\n" +
			"is\nover\nmultiple\nlines\n" +
			"ItemNumber2\n",
	},
	&Test{
		in: "{.repeated section pdata }\n" +
			"{item}\n" +
			"{.alternates with}\n" +
			"is\nover\nmultiple\nlines\n" +
			" {.end}\n",

		out: "ItemNumber1\n" +
			"is\nover\nmultiple\nlines\n" +
			"ItemNumber2\n",
	},
	&Test{
		in: "{.section pdata }\n" +
			"{.repeated section @ }\n" +
			"{item}={value}\n" +
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
		in: "{.repeated section vec }\n" +
			"{@}\n" +
			"{.end}\n",

		out: "elt1\n" +
			"elt2\n",
	},
	// Same but with a space before {.end}: was a bug.
	&Test{
		in: "{.repeated section vec }\n" +
			"{@} {.end}\n",

		out: "elt1 elt2 \n",
	},
	&Test{
		in: "{.repeated section integer}{.end}",

		err: "line 1: .repeated: cannot repeat integer (type int)",
	},

	// Nested names
	&Test{
		in: "{.section @ }\n" +
			"{innerT.item}={innerT.value}\n" +
			"{.end}",

		out: "ItemNumber1=ValueNumber1\n",
	},
	&Test{
		in: "{.section @ }\n" +
			"{innerT.item}={.section innerT}{.section value}{@}{.end}{.end}\n" +
			"{.end}",

		out: "ItemNumber1=ValueNumber1\n",
	},


	// Formatters
	&Test{
		in: "{.section pdata }\n" +
			"{header|uppercase}={integer|+1}\n" +
			"{header|html}={integer|str}\n" +
			"{.end}\n",

		out: "HEADER=78\n" +
			"Header=77\n",
	},

	&Test{
		in: "{raw}\n" +
			"{raw|html}\n",

		out: "&<>!@ #$%^\n" +
			"&amp;&lt;&gt;!@ #$%^\n",
	},

	&Test{
		in: "{.section emptystring}emptystring{.end}\n" +
			"{.section header}header{.end}\n",

		out: "\nheader\n",
	},

	&Test{
		in: "{.section true}1{.or}2{.end}\n" +
			"{.section false}3{.or}4{.end}\n",

		out: "1\n4\n",
	},

	&Test{
		in: "{bytes}",

		out: "hello",
	},

	// Maps

	&Test{
		in: "{mp.mapkey}\n",

		out: "Ahoy!\n",
	},
	&Test{
		in: "{innermap.mp.innerkey}\n",

		out: "55\n",
	},
	&Test{
		in: "{stringmap.stringkey1}\n",

		out: "stringresult\n",
	},
	&Test{
		in: "{.repeated section stringmap}\n" +
			"{@}\n" +
			"{.end}",

		out: "stringresult\n" +
			"stringresult\n",
	},

	// Interface values

	&Test{
		in: "{iface}",

		out: "[1 2 3]",
	},
	&Test{
		in: "{.repeated section iface}{@}{.alternates with} {.end}",

		out: "1 2 3",
	},
	&Test{
		in: "{.section iface}{@}{.end}",

		out: "[1 2 3]",
	},
}

func TestAll(t *testing.T) {
	s := new(S)
	// initialized by hand for clarity.
	s.header = "Header"
	s.integer = 77
	s.raw = "&<>!@ #$%^"
	s.innerT = t1
	s.data = []T{t1, t2}
	s.pdata = []*T{&t1, &t2}
	s.empty = []*T{}
	s.null = nil
	s.vec = new(vector.Vector)
	s.vec.Push("elt1")
	s.vec.Push("elt2")
	s.true = true
	s.false = false
	s.mp = make(map[string]string)
	s.mp["mapkey"] = "Ahoy!"
	s.innermap.mp = make(map[string]int)
	s.innermap.mp["innerkey"] = 55
	s.stringmap = make(map[string]string)
	s.stringmap["stringkey1"] = "stringresult" // the same value so repeated section is order-independent
	s.stringmap["stringkey2"] = "stringresult"
	s.bytes = strings.Bytes("hello")
	s.iface = []int{1, 2, 3}

	var buf bytes.Buffer
	for _, test := range tests {
		buf.Reset()
		tmpl, err := Parse(test.in, formatters)
		if err != nil {
			t.Error("unexpected parse error:", err)
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
	s.innerPointerT = &t1

	var buf bytes.Buffer
	input := "{.section @}{innerPointerT}{.end}"
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
