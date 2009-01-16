// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flag

/*
 * Flags
 *
 * Usage:
 *	1) Define flags using flag.String(), Bool(), Int(), etc. Example:
 *		import flag "flag"
 *		var ip *int = flag.Int("flagname", 1234, "help message for flagname")
 *	If you like, you can bind the flag to a variable using the Var() functions.
 *		var flagvar int
 *		func init() {
 *              	flag.IntVar(&flagvar, "flagname", 1234, "help message for flagname")
 *		}
 *	2) After all flags are defined, call
 *		flag.Parse()
 *	to parse the command line into the defined flags.
 *
 *	3) Flags may then be used directly. If you're using the flags themselves,
 *	they are all pointers; if you bind to variables, they're values.
 *		print("ip has value ", *ip, "\n");
 *		print("flagvar has value ", flagvar, "\n");
 *
 *	4) After parsing, flag.Arg(i) is the i'th argument after the flags.
 *	Args are indexed from 0 up to flag.NArg().
 *
 *	Command line flag syntax:
 *		-flag
 *		-flag=x
 *		-flag x
 *	One or two minus signs may be used; they are equivalent.
 *
 *	Flag parsing stops just before the first non-flag argument
 *	("-" is a non-flag argument) or after the terminator "--".
 *
 *	Integer flags accept 1234, 0664, 0x1234 and may be negative.
 *	Boolean flags may be 1, 0, t, f, true, false, TRUE, FALSE, True, False.
 */

import "fmt"

// BUG: ctoi, atoi, atob belong elsewhere
func ctoi(c int64) int64 {
	if '0' <= c && c <= '9' {
		return c - '0'
	}
	if 'a' <= c && c <= 'f' {
		return c - 'a'
	}
	if 'A' <= c && c <= 'F' {
		return c - 'A'
	}
	return 1000   // too large for any base
}

func atoi(s string) (value int64, ok bool) {
	if len(s) == 0 {
		return 0, false
	}
	if s[0] == '-' {
		n, t := atoi(s[1:len(s)]);
		return -n, t
	}
	var base int64 = 10;
	i := 0;
	if s[0] == '0' {
		base = 8;
		if len(s) > 1 && (s[1] == 'x' || s[1] == 'X') {
			base = 16;
			i += 2;
		}
	}
	var n int64 = 0;
	for ; i < len(s); i++ {
		k := ctoi(int64(s[i]));
		if k >= base {
			return 0, false
		}
		n = n * base + k
	}
	return n, true
}

func atob(str string) (value bool, ok bool) {
	switch str {
		case "1", "t", "T", "true", "TRUE", "True":
			return true, true;
		case "0", "f", "F", "false", "FALSE", "False":
			return false, true
	}
	return false, false
}

type (
	boolValue struct;
	intValue struct;
	int64Value struct;
	uintValue struct;
	uint64Value struct;
	stringValue struct;
)

// -- Bool Value
type boolValue struct {
	p *bool;
}

func newBoolValue(val bool, p *bool) *boolValue {
	*p = val;
	return &boolValue{p}
}

func (b *boolValue) set(val bool) {
	*b.p = val;
}

func (b *boolValue) str() string {
	return fmt.Sprintf("%v", *b.p)
}

// -- Int Value
type intValue struct {
	p	*int;
}

func newIntValue(val int, p *int) *intValue {
	*p = val;
	return &intValue{p}
}

func (i *intValue) set(val int) {
	*i.p = val;
}

func (i *intValue) str() string {
	return fmt.Sprintf("%v", *i.p)
}

// -- Int64 Value
type int64Value struct {
	p	*int64;
}

func newInt64Value(val int64, p *int64) *int64Value {
	*p = val;
	return &int64Value{p}
}

func (i *int64Value) set(val int64) {
	*i.p = val;
}

func (i *int64Value) str() string {
	return fmt.Sprintf("%v", *i.p)
}

// -- Uint Value
type uintValue struct {
	p	*uint;
}

func newUintValue(val uint, p *uint) *uintValue {
	*p = val;
	return &uintValue{p}
}

func (i *uintValue) set(val uint) {
	*i.p = val
}

func (i *uintValue) str() string {
	return fmt.Sprintf("%v", *i.p)
}

// -- uint64 Value
type uint64Value struct {
	p	*uint64;
}

func newUint64Value(val uint64, p *uint64) *uint64Value {
	*p = val;
	return &uint64Value{p}
}

func (i *uint64Value) set(val uint64) {
	*i.p = val;
}

func (i *uint64Value) str() string {
	return fmt.Sprintf("%v", *i.p)
}

// -- string Value
type stringValue struct {
	p	*string;
}

func newStringValue(val string, p *string) *stringValue {
	*p = val;
	return &stringValue{p}
}

func (s *stringValue) set(val string) {
	*s.p = val;
}

func (s *stringValue) str() string {
	return fmt.Sprintf("%#q", *s.p)
}

// -- Value interface
type Value interface {
	str() string;
}

// -- Flag structure (internal)
export type Flag struct {
	name	string;
	usage	string;
	value	Value;
}

type allFlags struct {
	actual map[string] *Flag;
	formal map[string] *Flag;
	first_arg	int;
}


func New() *allFlags {
	f := new(allFlags);
	f.first_arg = 1;	// 0 is the program name, 1 is first arg
	f.actual = make(map[string] *Flag);
	f.formal = make(map[string] *Flag);
	return f;
}

var flags *allFlags = New();

export func PrintDefaults() {
	for k, f := range flags.formal {
		print("  -", f.name, "=", f.value.str(), ": ", f.usage, "\n");
	}
}

export func Usage() {
	if sys.argc() > 0 {
		print("Usage of ", sys.argv(0), ": \n");
	} else {
		print("Usage: \n");
	}
	PrintDefaults();
	sys.exit(1);
}

export func NFlag() int {
	return len(flags.actual)
}

export func Arg(i int) string {
	i += flags.first_arg;
	if i < 0 || i >= sys.argc() {
		return "";
	}
	return sys.argv(i)
}

export func NArg() int {
	return sys.argc() - flags.first_arg
}

func add(name string, value Value, usage string) {
	f := new(Flag);
	f.name = name;
	f.usage = usage;
	f.value = value;
	dummy, alreadythere := flags.formal[name];
	if alreadythere {
		print("flag redefined: ", name, "\n");
		panic("flag redefinition");
	}
	flags.formal[name] = f;
}

export func Bool(name string, value bool, usage string) *bool {
	p := new(bool);
	add(name, newBoolValue(value, p), usage);
	return p;
}

export func BoolVar(p *bool, name string, value bool, usage string) {
	add(name, newBoolValue(value, p), usage);
}

export func Int(name string, value int, usage string) *int {
	p := new(int);
	add(name, newIntValue(value, p), usage);
	return p;
}

export func IntVar(p *int, name string, value int, usage string) {
	add(name, newIntValue(value, p), usage);
}

export func Int64(name string, value int64, usage string) *int64 {
	p := new(int64);
	add(name, newInt64Value(value, p), usage);
	return p;
}

export func Int64Var(p *int64, name string, value int64, usage string) {
	add(name, newInt64Value(value, p), usage);
}

export func Uint(name string, value uint, usage string) *uint {
	p := new(uint);
	add(name, newUintValue(value, p), usage);
	return p;
}

export func UintVar(p *uint, name string, value uint, usage string) {
	add(name, newUintValue(value, p), usage);
}

export func Uint64(name string, value uint64, usage string) *uint64 {
	p := new(uint64);
	add(name, newUint64Value(value, p), usage);
	return p;
}

export func Uint64Var(p *uint64, name string, value uint64, usage string) {
	add(name, newUint64Value(value, p), usage);
}

export func String(name, value string, usage string) *string {
	p := new(string);
	add(name, newStringValue(value, p), usage);
	return p;
}

export func StringVar(p *string, name, value string, usage string) {
	add(name, newStringValue(value, p), usage);
}

func (f *allFlags) ParseOne(index int) (ok bool, next int)
{
	s := sys.argv(index);
	f.first_arg = index;  // until proven otherwise
	if len(s) == 0 {
		return false, -1
	}
	if s[0] != '-' {
		return false, -1
	}
	num_minuses := 1;
	if len(s) == 1 {
		return false, index
	}
	if s[1] == '-' {
		num_minuses++;
		if len(s) == 2 {	// "--" terminates the flags
			return false, index + 1
		}
	}
	name := s[num_minuses : len(s)];
	if len(name) == 0 || name[0] == '-' || name[0] == '=' {
		print("bad flag syntax: ", s, "\n");
		Usage();
	}

	// it's a flag. does it have an argument?
	has_value := false;
	value := "";
	for i := 1; i < len(name); i++ {  // equals cannot be first
		if name[i] == '=' {
			value = name[i+1 : len(name)];
			has_value = true;
			name = name[0 : i];
			break;
		}
	}
	flag, alreadythere := flags.actual[name];
	if alreadythere {
		print("flag specified twice: -", name, "\n");
		Usage();
	}
	m := flags.formal;
	flag, alreadythere = m[name]; // BUG
	if !alreadythere {
		print("flag provided but not defined: -", name, "\n");
		Usage();
	}
	if f, ok := flag.value.(*boolValue); ok {
		if has_value {
			k, ok := atob(value);
			if !ok {
				print("invalid boolean value ", value, " for flag: -", name, "\n");
				Usage();
			}
			f.set(k)
		} else {
			f.set(true)
		}
	} else {
		// It must have a value, which might be the next argument.
		if !has_value && index < sys.argc()-1 {
			// value is the next arg
			has_value = true;
			index++;
			value = sys.argv(index);
		}
		if !has_value {
			print("flag needs an argument: -", name, "\n");
			Usage();
		}
		if f, ok := flag.value.(*stringValue); ok {
			f.set(value)
		} else {
			// It's an integer flag.  TODO(r): check for overflow?
			k, ok := atoi(value);
			if !ok {
				print("invalid integer value ", value, " for flag: -", name, "\n");
				Usage();
			}
			if f, ok := flag.value.(*intValue); ok {
				f.set(int(k));
			} else if f, ok := flag.value.(*int64Value); ok {
				f.set(k);
			} else if f, ok := flag.value.(*uintValue); ok {
				f.set(uint(k));
			} else if f, ok := flag.value.(*uint64Value); ok {
				f.set(uint64(k));
			}
		}
	}
	flags.actual[name] = flag;
	return true, index + 1
}

export func Parse() {
	for i := 1; i < sys.argc(); {
		ok, next := flags.ParseOne(i);
		if next > 0 {
			flags.first_arg = next;
			i = next;
		}
		if !ok {
			break
		}
	}
}
