// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt";
	"io";
	"os";
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
		err := Execute(test.in, s, &buf);
		if err != nil {
			t.Error("unexpected error:", err)
		}
		if string(buf.Data()) != test.out {
			t.Errorf("for %q: expected %q got %q", test.in, test.out, string(buf.Data()));
		}
	}
}

/*
func TestParser(t *testing.T) {
	t1 := &T{ "ItemNumber1", "ValueNumber1" };
	t2 := &T{ "ItemNumber2", "ValueNumber2" };
	a := []*T{ t1, t2 };
	s := &S{ "Header", 77, a };
	err := Execute(
		"{#hello world}\n"
		"some text: {.meta-left}{.space}{.meta-right}\n"
		"{.meta-left}\n"
		"{.meta-right}\n"
		"{.section data }\n"
		"some text for the section\n"
		"{header} for iteration number {integer}\n"
		"	{.repeated section @}\n"
		"repeated section: {value1}={value2}\n"
		"	{.end}\n"
		"{.or}\n"
		"This appears only if there is no data\n"
		"{.end }\n"
		"this is the end\n"
		, s, os.Stdout);
	if err != nil {
		t.Error(err)
	}
}
*/

func TestBadDriverType(t *testing.T) {
	err := Execute("hi", "hello", os.Stdout);
	if err == nil {
		t.Error("failed to detect string as driver type")
	}
	var s S;
}
