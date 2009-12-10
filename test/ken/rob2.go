// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


package main

const nilchar = 0;

type Atom struct {
	str		string;
	integer		int;
	next		*Slist;	/* in hash bucket */
}

type List struct {
	car		*Slist;
	cdr*Slist;
}

type Slist struct {
	isatom		bool;
	isstring	bool;
	//union {
	atom		Atom;
	list		List;
	//} u;

}

func (this *Slist) Car() *Slist {
	return this.list.car;
}

func (this *Slist) Cdr() *Slist {
	return this.list.cdr;
}

func (this *Slist) String() string {
	return this.atom.str;
}

func (this *Slist) Integer() int {
	return this.atom.integer;
}

func (slist *Slist) Free() {
	if slist == nil {
		return;
	}
	if slist.isatom {
//		free(slist.String());
	} else {
		slist.Car().Free();
		slist.Cdr().Free();
	}
//	free(slist);
}

//Slist* atom(byte *s, int i);

var token int;
var peekc int = -1;
var lineno int32 = 1;

var input string;
var inputindex int = 0;
var tokenbuf [100]byte;
var tokenlen int = 0;

const EOF int = -1;

func main() {
	var list *Slist;

	OpenFile();
	for ;; {
		list = Parse();
		if list == nil {
			break;
		}
		list.Print();
		list.Free();
		break;
	}
}

func (slist *Slist) PrintOne(doparen bool) {
	if slist == nil {
		return;
	}
	if slist.isatom {
		if slist.isstring {
			print(slist.String());
		} else {
			print(slist.Integer());
		}
	} else {
		if doparen {
			print("(" );
		}
		slist.Car().PrintOne(true);
		if slist.Cdr() != nil {
			print(" ");
			slist.Cdr().PrintOne(false);
		}
		if doparen {
			print(")");
		}
	}
}

func (slist *Slist) Print() {
	slist.PrintOne(true);
	print("\n");
}

func Get() int {
	var c int;

	if peekc >= 0 {
		c = peekc;
		peekc = -1;
	} else {
		c = int(input[inputindex]);
		inputindex++;
		if c == '\n' {
			lineno = lineno + 1;
		}
		if c == nilchar {
			inputindex = inputindex - 1;
			c = EOF;
		}
	}
	return c;
}

func WhiteSpace(c int) bool {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

func NextToken() {
	var i, c int;

	tokenbuf[0] = nilchar;	// clear previous token
	c = Get();
	for WhiteSpace(c) {
		c = Get();
	}
	switch c {
	case EOF:
		token = EOF;
	case '(', ')':
		token = c;
		break;
	default:
		for i = 0; i < 100 - 1; {	// sizeof tokenbuf - 1
			tokenbuf[i] = byte(c);
			i = i + 1;
			c = Get();
			if c == EOF {
				break;
			}
			if WhiteSpace(c) || c == ')' {
				peekc = c;
				break;
			}
		}
		if i >= 100 - 1 {	// sizeof tokenbuf - 1
			panic("atom too long\n");
		}
		tokenlen = i;
		tokenbuf[i] = nilchar;
		if '0' <= tokenbuf[0] && tokenbuf[0] <= '9' {
			token = '0';
		} else {
			token = 'A';
		}
	}
}

func Expect(c int) {
	if token != c {
		print("parse error: expected ", c, "\n");
		panic("parse");
	}
	NextToken();
}

// Parse a non-parenthesized list up to a closing paren or EOF
func ParseList() *Slist {
	var slist, retval *Slist;

	slist = new(Slist);
	slist.list.car = nil;
	slist.list.cdr = nil;
	slist.isatom = false;
	slist.isstring = false;

	retval = slist;
	for ;; {
		slist.list.car = Parse();
		if token == ')' || token == EOF {	// empty cdr
			break;
		}
		slist.list.cdr = new(Slist);
		slist = slist.list.cdr;
	}
	return retval;
}

func atom(i int) *Slist	{ // BUG: uses tokenbuf; should take argument)
	var slist *Slist;

	slist = new(Slist);
	if token == '0' {
		slist.atom.integer = i;
		slist.isstring = false;
	} else {
		slist.atom.str = string(tokenbuf[0:tokenlen]);
		slist.isstring = true;
	}
	slist.isatom = true;
	return slist;
}

func atoi() int	{ // BUG: uses tokenbuf; should take argument)
	var v int = 0;
	for i := 0; i < tokenlen && '0' <= tokenbuf[i] && tokenbuf[i] <= '9'; i = i + 1 {
		v = 10 * v + int(tokenbuf[i] - '0');
	}
	return v;
}

func Parse() *Slist {
	var slist *Slist;

	if token == EOF || token == ')' {
		return nil;
	}
	if token == '(' {
		NextToken();
		slist = ParseList();
		Expect(')');
		return slist;
	} else {
		// Atom
		switch token {
		case EOF:
			return nil;
		case '0':
			slist = atom(atoi());
		case '"', 'A':
			slist = atom(0);
		default:
			slist = nil;
			print("unknown token: ", token, "\n");
		}
		NextToken();
		return slist;
	}
	return nil;
}

func OpenFile() {
	input = "(defn foo (add 12 34))\n\x00";
	inputindex = 0;
	peekc = -1;		// BUG
	NextToken();
}
