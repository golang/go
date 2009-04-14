// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt";
	"io";
	"os";
	"reflect";
	"template";
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
	data []T;
	pdata []*T;
	empty []*T;
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

func writer(f func(interface{}) string) (func(io.Write, interface{}, string)) {
	return func(w io.Write, v interface{}, format string) {
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
	&Test{ "     {#comment}   \n", "" },

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

	// Formatters
	&Test{
		"{.section pdata }\n"
		"{header|uppercase}={integer|+1}\n"
		"{header|html}={integer|str}\n"
		"{.end}\n",

		"HEADER=78\n"
		"Header=77\n"
	},

}

func TestAll(t *testing.T) {
	s := new(S);
	// initialized by hand for clarity.
	s.header = "Header";
	s.integer = 77;
	s.data = []T{ t1, t2 };
	s.pdata = []*T{ &t1, &t2 };
	s.empty = []*T{ };
	s.null = nil;

	var buf io.ByteBuffer;
	for i, test := range tests {
		buf.Reset();
		tmpl, err, line := Parse(test.in, formatters);
		if err != nil {
			t.Error("unexpected parse error:", err, "line", line);
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
	tmpl, err, line := Parse("template: {@}", nil);
	if err != nil {
		t.Error("unexpected parse error:", err)
	}
	var b io.ByteBuffer;
	err = tmpl.Execute("hello", &b);
	s := string(b.Data());
	if s != "template: hello" {
		t.Errorf("failed passing string as data: expected %q got %q", "template: hello", s)
	}
}
