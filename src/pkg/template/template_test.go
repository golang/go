// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes";
	"fmt";
	"io";
	"os";
	"reflect";
	"testing";
)

type Test struct {
	in, out string
}

type T struct {
	item string;
	value string;
}

type S struct {
	header string;
	integer int;
	raw string;
	innerT T;
	innerPointerT *T;
	data []T;
	pdata []*T;
	empty []*T;
	emptystring string;
	null []*T;
}

var t1 = T{ "ItemNumber1", "ValueNumber1" }
var t2 = T{ "ItemNumber2", "ValueNumber2" }

func uppercase(v interface{}) string {
	s := v.(string);
	t := "";
	for i := 0; i < len(s); i++ {
		c := s[i];
		if 'a' <= c && c <= 'z' {
			c = c + 'A' - 'a'
		}
		t += string(c);
	}
	return t;
}

func plus1(v interface{}) string {
	i := v.(int);
	return fmt.Sprint(i + 1);
}

func writer(f func(interface{}) string) (func(io.Writer, interface{}, string)) {
	return func(w io.Writer, v interface{}, format string) {
		io.WriteString(w, f(v));
	}
}


var formatters = FormatterMap {
	"uppercase" : writer(uppercase),
	"+1" : writer(plus1),
}

var tests = []*Test {
	// Simple
	&Test{ "", "" },
	&Test{ "abc\ndef\n", "abc\ndef\n" },
	&Test{ " {.meta-left}   \n", "{" },
	&Test{ " {.meta-right}   \n", "}" },
	&Test{ " {.space}   \n", " " },
	&Test{ " {.tab}   \n", "\t" },
	&Test{ "     {#comment}   \n", "" },

	// Variables at top level
	&Test{
		"{header}={integer}\n",

		"Header=77\n"
	},

	// Section
	&Test{
		"{.section data }\n"
		"some text for the section\n"
		"{.end}\n",

		"some text for the section\n"
	},
	&Test{
		"{.section data }\n"
		"{header}={integer}\n"
		"{.end}\n",

		"Header=77\n"
	},
	&Test{
		"{.section pdata }\n"
		"{header}={integer}\n"
		"{.end}\n",

		"Header=77\n"
	},
	&Test{
		"{.section pdata }\n"
		"data present\n"
		"{.or}\n"
		"data not present\n"
		"{.end}\n",

		"data present\n"
	},
	&Test{
		"{.section empty }\n"
		"data present\n"
		"{.or}\n"
		"data not present\n"
		"{.end}\n",

		"data not present\n"
	},
	&Test{
		"{.section null }\n"
		"data present\n"
		"{.or}\n"
		"data not present\n"
		"{.end}\n",

		"data not present\n"
	},
	&Test{
		"{.section pdata }\n"
		"{header}={integer}\n"
		"{.section @ }\n"
		"{header}={integer}\n"
		"{.end}\n"
		"{.end}\n",

		"Header=77\n"
		"Header=77\n"
	},
	&Test{
		"{.section data}{.end} {header}\n",

		" Header\n"
	},

	// Repeated
	&Test{
		"{.section pdata }\n"
		"{.repeated section @ }\n"
		"{item}={value}\n"
		"{.end}\n"
		"{.end}\n",

		"ItemNumber1=ValueNumber1\n"
		"ItemNumber2=ValueNumber2\n"
	},
	&Test{
		"{.section pdata }\n"
		"{.repeated section @ }\n"
		"{item}={value}\n"
		"{.or}\n"
		"this should not appear\n"
		"{.end}\n"
		"{.end}\n",

		"ItemNumber1=ValueNumber1\n"
		"ItemNumber2=ValueNumber2\n"
	},
	&Test{
		"{.section @ }\n"
		"{.repeated section empty }\n"
		"{item}={value}\n"
		"{.or}\n"
		"this should appear: empty field\n"
		"{.end}\n"
		"{.end}\n",

		"this should appear: empty field\n"
	},
	&Test{
		"{.section pdata }\n"
		"{.repeated section @ }\n"
		"{item}={value}\n"
		"{.alternates with}DIVIDER\n"
		"{.or}\n"
		"this should not appear\n"
		"{.end}\n"
		"{.end}\n",

		"ItemNumber1=ValueNumber1\n"
		"DIVIDER\n"
		"ItemNumber2=ValueNumber2\n"
	},

	// Nested names
	&Test{
		"{.section @ }\n"
		"{innerT.item}={innerT.value}\n"
		"{.end}",

		"ItemNumber1=ValueNumber1\n"
	},

	// Formatters
	&Test{
		"{.section pdata }\n"
		"{header|uppercase}={integer|+1}\n"
		"{header|html}={integer|str}\n"
		"{.end}\n",

		"HEADER=78\n"
		"Header=77\n"
	},

	&Test{
		"{raw}\n"
		"{raw|html}\n",

		"&<>!@ #$%^\n"
		"&amp;&lt;&gt;!@ #$%^\n"
	},

	&Test{
		"{.section emptystring}emptystring{.end}\n"
		"{.section header}header{.end}\n",

		"\nheader\n"
	},
}

func TestAll(t *testing.T) {
	s := new(S);
	// initialized by hand for clarity.
	s.header = "Header";
	s.integer = 77;
	s.raw = "&<>!@ #$%^";
	s.innerT = t1;
	s.data = []T{ t1, t2 };
	s.pdata = []*T{ &t1, &t2 };
	s.empty = []*T{ };
	s.null = nil;

	var buf bytes.Buffer;
	for i, test := range tests {
		buf.Reset();
		tmpl, err := Parse(test.in, formatters);
		if err != nil {
			t.Error("unexpected parse error:", err);
			continue;
		}
		err = tmpl.Execute(s, &buf);
		if err != nil {
			t.Error("unexpected execute error:", err)
		}
		if string(buf.Data()) != test.out {
			t.Errorf("for %q: expected %q got %q", test.in, test.out, string(buf.Data()));
		}
	}
}

func TestStringDriverType(t *testing.T) {
	tmpl, err := Parse("template: {@}", nil);
	if err != nil {
		t.Error("unexpected parse error:", err)
	}
	var b bytes.Buffer;
	err = tmpl.Execute("hello", &b);
	if err != nil {
		t.Error("unexpected execute error:", err)
	}
	s := string(b.Data());
	if s != "template: hello" {
		t.Errorf("failed passing string as data: expected %q got %q", "template: hello", s)
	}
}

func TestTwice(t *testing.T) {
	tmpl, err := Parse("template: {@}", nil);
	if err != nil {
		t.Error("unexpected parse error:", err)
	}
	var b bytes.Buffer;
	err = tmpl.Execute("hello", &b);
	if err != nil {
		t.Error("unexpected parse error:", err)
	}
	s := string(b.Data());
	text := "template: hello";
	if s != text {
		t.Errorf("failed passing string as data: expected %q got %q", text, s);
	}
	err = tmpl.Execute("hello", &b);
	if err != nil {
		t.Error("unexpected parse error:", err)
	}
	s = string(b.Data());
	text += text;
	if s != text {
		t.Errorf("failed passing string as data: expected %q got %q", text, s);
	}
}

func TestCustomDelims(t *testing.T) {
	// try various lengths.  zero should catch error.
	for i := 0; i < 7; i++ {
		for j := 0; j < 7; j++ {
			tmpl := New(nil);
			// first two chars deliberately the same to test equal left and right delims
			ldelim := "$!#$%^&"[0:i];
			rdelim := "$*&^%$!"[0:j];
			tmpl.SetDelims(ldelim, rdelim);
			// if braces, this would be template: {@}{.meta-left}{.meta-right}
			text := "template: " +
				ldelim + "@" + rdelim +
				ldelim + ".meta-left" + rdelim +
				ldelim + ".meta-right" + rdelim;
			err := tmpl.Parse(text);
			if err != nil {
				if i == 0 || j == 0 {	// expected
					continue
				}
				t.Error("unexpected parse error:", err)
			} else if i == 0 || j == 0 {
				t.Errorf("expected parse error for empty delimiter: %d %d %q %q", i, j, ldelim, rdelim);
				continue;
			}
			var b bytes.Buffer;
			err = tmpl.Execute("hello", &b);
			s := string(b.Data());
			if s != "template: hello" + ldelim + rdelim {
				t.Errorf("failed delim check(%q %q) %q got %q", ldelim, rdelim, text, s)
			}
		}
	}
}

// Test that a variable evaluates to the field itself and does not further indirection
func TestVarIndirection(t *testing.T) {
	s := new(S);
	// initialized by hand for clarity.
	s.innerPointerT = &t1;

	var buf bytes.Buffer;
	input := "{.section @}{innerPointerT}{.end}";
	tmpl, err := Parse(input, nil);
	if err != nil {
		t.Fatal("unexpected parse error:", err);
	}
	err = tmpl.Execute(s, &buf);
	if err != nil {
		t.Fatal("unexpected execute error:", err)
	}
	expect := fmt.Sprintf("%v", &t1);	// output should be hex address of t1
	if string(buf.Data()) != expect {
		t.Errorf("for %q: expected %q got %q", input, expect, string(buf.Data()));
	}
}
