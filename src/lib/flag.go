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
 *			flag.IntVar(&flagvar, "flagname", 1234, "help message for flagname")
 *		}
 *
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

import (
	"fmt";
	"strconv"
)

// BUG: atob belongs elsewhere
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

func (b *boolValue) set(s string) bool {
	v, ok  := atob(s);
	*b.p = v;
	return ok
}

func (b *boolValue) String() string {
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

func (i *intValue) set(s string) bool {
	v, err  := strconv.Atoi(s);
	*i.p = int(v);
	return err == nil
}

func (i *intValue) String() string {
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

func (i *int64Value) set(s string) bool {
	v, err  := strconv.Atoi64(s);
	*i.p = v;
	return err == nil;
}

func (i *int64Value) String() string {
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

func (i *uintValue) set(s string) bool {
	v, err  := strconv.Atoui(s);
	*i.p = uint(v);
	return err == nil;
}

func (i *uintValue) String() string {
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

func (i *uint64Value) set(s string) bool {
	v, err := strconv.Atoui64(s);
	*i.p = uint64(v);
	return err == nil;
}

func (i *uint64Value) String() string {
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

func (s *stringValue) set(val string) bool {
	*s.p = val;
	return true;
}

func (s *stringValue) String() string {
	return fmt.Sprintf("%s", *s.p)
}

// -- FlagValue interface
type FlagValue interface {
	String() string;
	set(string) bool;
}

// -- Flag structure
type Flag struct {
	Name	string;	// name as it appears on command line
	Usage	string;	// help message
	Value	FlagValue;	// value as set
	DefValue	string;	// default value (as text); for usage message
}

type allFlags struct {
	actual map[string] *Flag;
	formal map[string] *Flag;
	first_arg	int;	// 0 is the program name, 1 is first arg
}

var flags *allFlags = &allFlags{make(map[string] *Flag), make(map[string] *Flag), 1}

// Visit all flags, including those defined but not set.
func VisitAll(fn func(*Flag)) {
	for k, f := range flags.formal {
		fn(f)
	}
}

// Visit only those flags that have been set
func Visit(fn func(*Flag)) {
	for k, f := range flags.actual {
		fn(f)
	}
}

func Lookup(name string) *Flag {
	f, ok := flags.formal[name];
	if !ok {
		return nil
	}
	return f
}

func Set(name, value string) bool {
	f, ok := flags.formal[name];
	if !ok {
		return false
	}
	ok = f.Value.set(value);
	if !ok {
		return false
	}
	flags.actual[name] = f;
	return true;
}

func PrintDefaults() {
	VisitAll(func(f *Flag) {
		format := "  -%s=%s: %s\n";
		if s, ok := f.Value.(*stringValue); ok {
			// put quotes on the value
			format = "  -%s=%q: %s\n";
		}
		fmt.Printf(format, f.Name, f.DefValue, f.Usage);
	})
}

func Usage() {
	if len(sys.Args) > 0 {
		print("Usage of ", sys.Args[0], ": \n");
	} else {
		print("Usage: \n");
	}
	PrintDefaults();
	sys.Exit(1);
}

func NFlag() int {
	return len(flags.actual)
}

func Arg(i int) string {
	i += flags.first_arg;
	if i < 0 || i >= len(sys.Args) {
		return "";
	}
	return sys.Args[i]
}

func NArg() int {
	return len(sys.Args) - flags.first_arg
}

func add(name string, value FlagValue, usage string) {
	// Remember the default value as a string; it won't change.
	f := &Flag{name, usage, value, value.String()};
	dummy, alreadythere := flags.formal[name];
	if alreadythere {
		print("flag redefined: ", name, "\n");
		panic("flag redefinition");	// Happens only if flags are declared with identical names
	}
	flags.formal[name] = f;
}

func BoolVar(p *bool, name string, value bool, usage string) {
	add(name, newBoolValue(value, p), usage);
}

func Bool(name string, value bool, usage string) *bool {
	p := new(bool);
	BoolVar(p, name, value, usage);
	return p;
}

func IntVar(p *int, name string, value int, usage string) {
	add(name, newIntValue(value, p), usage);
}

func Int(name string, value int, usage string) *int {
	p := new(int);
	IntVar(p, name, value, usage);
	return p;
}

func Int64Var(p *int64, name string, value int64, usage string) {
	add(name, newInt64Value(value, p), usage);
}

func Int64(name string, value int64, usage string) *int64 {
	p := new(int64);
	Int64Var(p, name, value, usage);
	return p;
}

func UintVar(p *uint, name string, value uint, usage string) {
	add(name, newUintValue(value, p), usage);
}

func Uint(name string, value uint, usage string) *uint {
	p := new(uint);
	UintVar(p, name, value, usage);
	return p;
}

func Uint64Var(p *uint64, name string, value uint64, usage string) {
	add(name, newUint64Value(value, p), usage);
}

func Uint64(name string, value uint64, usage string) *uint64 {
	p := new(uint64);
	Uint64Var(p, name, value, usage);
	return p;
}

func StringVar(p *string, name, value string, usage string) {
	add(name, newStringValue(value, p), usage);
}

func String(name, value string, usage string) *string {
	p := new(string);
	StringVar(p, name, value, usage);
	return p;
}

func (f *allFlags) parseOne(index int) (ok bool, next int)
{
	s := sys.Args[index];
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
	if f, ok := flag.Value.(*boolValue); ok {	// special case: doesn't need an arg
		if has_value {
			if !f.set(value) {
				print("invalid boolean value ", value, " for flag: -", name, "\n");
				Usage();
			}
		} else {
			f.set("true")
		}
	} else {
		// It must have a value, which might be the next argument.
		if !has_value && index < len(sys.Args)-1 {
			// value is the next arg
			has_value = true;
			index++;
			value = sys.Args[index];
		}
		if !has_value {
			print("flag needs an argument: -", name, "\n");
			Usage();
		}
		ok = flag.Value.set(value);
		if !ok {
			print("invalid value ", value, " for flag: -", name, "\n");
				Usage();
		}
	}
	flags.actual[name] = flag;
	return true, index + 1
}

func Parse() {
	for i := 1; i < len(sys.Args); {
		ok, next := flags.parseOne(i);
		if next > 0 {
			flags.first_arg = next;
			i = next;
		}
		if !ok {
			break
		}
	}
}
