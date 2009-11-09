// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	The flag package implements command-line flag parsing.

	Usage:

	1) Define flags using flag.String(), Bool(), Int(), etc. Example:
		import "flag"
		var ip *int = flag.Int("flagname", 1234, "help message for flagname")
	If you like, you can bind the flag to a variable using the Var() functions.
		var flagvar int
		func init() {
			flag.IntVar(&flagvar, "flagname", 1234, "help message for flagname")
		}

	2) After all flags are defined, call
		flag.Parse()
	to parse the command line into the defined flags.

	3) Flags may then be used directly. If you're using the flags themselves,
	they are all pointers; if you bind to variables, they're values.
		fmt.Println("ip has value ", *ip);
		fmt.Println("flagvar has value ", flagvar);

	4) After parsing, flag.Arg(i) is the i'th argument after the flags.
	Args are indexed from 0 up to flag.NArg().

	Command line flag syntax:
		-flag
		-flag=x
		-flag x
	One or two minus signs may be used; they are equivalent.

	Flag parsing stops just before the first non-flag argument
	("-" is a non-flag argument) or after the terminator "--".

	Integer flags accept 1234, 0664, 0x1234 and may be negative.
	Boolean flags may be 1, 0, t, f, true, false, TRUE, FALSE, True, False.
*/
package flag

import (
	"fmt";
	"os";
	"strconv";
)

// TODO(r): BUG: atob belongs elsewhere
func atob(str string) (value bool, ok bool) {
	switch str {
	case "1", "t", "T", "true", "TRUE", "True":
		return true, true
	case "0", "f", "F", "false", "FALSE", "False":
		return false, true
	}
	return false, false;
}

// -- Bool Value
type boolValue struct {
	p *bool;
}

func newBoolValue(val bool, p *bool) *boolValue {
	*p = val;
	return &boolValue{p};
}

func (b *boolValue) set(s string) bool {
	v, ok := atob(s);
	*b.p = v;
	return ok;
}

func (b *boolValue) String() string	{ return fmt.Sprintf("%v", *b.p) }

// -- Int Value
type intValue struct {
	p *int;
}

func newIntValue(val int, p *int) *intValue {
	*p = val;
	return &intValue{p};
}

func (i *intValue) set(s string) bool {
	v, err := strconv.Atoi(s);
	*i.p = int(v);
	return err == nil;
}

func (i *intValue) String() string	{ return fmt.Sprintf("%v", *i.p) }

// -- Int64 Value
type int64Value struct {
	p *int64;
}

func newInt64Value(val int64, p *int64) *int64Value {
	*p = val;
	return &int64Value{p};
}

func (i *int64Value) set(s string) bool {
	v, err := strconv.Atoi64(s);
	*i.p = v;
	return err == nil;
}

func (i *int64Value) String() string	{ return fmt.Sprintf("%v", *i.p) }

// -- Uint Value
type uintValue struct {
	p *uint;
}

func newUintValue(val uint, p *uint) *uintValue {
	*p = val;
	return &uintValue{p};
}

func (i *uintValue) set(s string) bool {
	v, err := strconv.Atoui(s);
	*i.p = uint(v);
	return err == nil;
}

func (i *uintValue) String() string	{ return fmt.Sprintf("%v", *i.p) }

// -- uint64 Value
type uint64Value struct {
	p *uint64;
}

func newUint64Value(val uint64, p *uint64) *uint64Value {
	*p = val;
	return &uint64Value{p};
}

func (i *uint64Value) set(s string) bool {
	v, err := strconv.Atoui64(s);
	*i.p = uint64(v);
	return err == nil;
}

func (i *uint64Value) String() string	{ return fmt.Sprintf("%v", *i.p) }

// -- string Value
type stringValue struct {
	p *string;
}

func newStringValue(val string, p *string) *stringValue {
	*p = val;
	return &stringValue{p};
}

func (s *stringValue) set(val string) bool {
	*s.p = val;
	return true;
}

func (s *stringValue) String() string	{ return fmt.Sprintf("%s", *s.p) }

// -- Float Value
type floatValue struct {
	p *float;
}

func newFloatValue(val float, p *float) *floatValue {
	*p = val;
	return &floatValue{p};
}

func (f *floatValue) set(s string) bool {
	v, err := strconv.Atof(s);
	*f.p = v;
	return err == nil;
}

func (f *floatValue) String() string	{ return fmt.Sprintf("%v", *f.p) }

// -- Float64 Value
type float64Value struct {
	p *float64;
}

func newFloat64Value(val float64, p *float64) *float64Value {
	*p = val;
	return &float64Value{p};
}

func (f *float64Value) set(s string) bool {
	v, err := strconv.Atof64(s);
	*f.p = v;
	return err == nil;
}

func (f *float64Value) String() string	{ return fmt.Sprintf("%v", *f.p) }

// FlagValue is the interface to the dynamic value stored in a flag.
// (The default value is represented as a string.)
type FlagValue interface {
	String() string;
	set(string) bool;
}

// A Flag represents the state of a flag.
type Flag struct {
	Name		string;		// name as it appears on command line
	Usage		string;		// help message
	Value		FlagValue;	// value as set
	DefValue	string;		// default value (as text); for usage message
}

type allFlags struct {
	actual		map[string]*Flag;
	formal		map[string]*Flag;
	first_arg	int;	// 0 is the program name, 1 is first arg
}

var flags *allFlags = &allFlags{make(map[string]*Flag), make(map[string]*Flag), 1}

// VisitAll visits the flags, calling fn for each. It visits all flags, even those not set.
func VisitAll(fn func(*Flag)) {
	for _, f := range flags.formal {
		fn(f)
	}
}

// Visit visits the flags, calling fn for each. It visits only those flags that have been set.
func Visit(fn func(*Flag)) {
	for _, f := range flags.actual {
		fn(f)
	}
}

// Lookup returns the Flag structure of the named flag, returning nil if none exists.
func Lookup(name string) *Flag {
	f, ok := flags.formal[name];
	if !ok {
		return nil
	}
	return f;
}

// Set sets the value of the named flag.  It returns true if the set succeeded; false if
// there is no such flag defined.
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

// PrintDefaults prints to standard error the default values of all defined flags.
func PrintDefaults() {
	VisitAll(func(f *Flag) {
		format := "  -%s=%s: %s\n";
		if _, ok := f.Value.(*stringValue); ok {
			// put quotes on the value
			format = "  -%s=%q: %s\n"
		}
		fmt.Fprintf(os.Stderr, format, f.Name, f.DefValue, f.Usage);
	})
}

// Usage prints to standard error a default usage message documenting all defined flags.
// The function is a variable that may be changed to point to a custom function.
var Usage = func() {
	fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0]);
	PrintDefaults();
}

func NFlag() int	{ return len(flags.actual) }

// Arg returns the i'th command-line argument.  Arg(0) is the first remaining argument
// after flags have been processed.
func Arg(i int) string {
	i += flags.first_arg;
	if i < 0 || i >= len(os.Args) {
		return ""
	}
	return os.Args[i];
}

// NArg is the number of arguments remaining after flags have been processed.
func NArg() int	{ return len(os.Args) - flags.first_arg }

// Args returns the non-flag command-line arguments.
func Args() []string	{ return os.Args[flags.first_arg : len(os.Args)] }

func add(name string, value FlagValue, usage string) {
	// Remember the default value as a string; it won't change.
	f := &Flag{name, usage, value, value.String()};
	_, alreadythere := flags.formal[name];
	if alreadythere {
		fmt.Fprintln(os.Stderr, "flag redefined:", name);
		panic("flag redefinition");	// Happens only if flags are declared with identical names
	}
	flags.formal[name] = f;
}

// BoolVar defines a bool flag with specified name, default value, and usage string.
// The argument p points to a bool variable in which to store the value of the flag.
func BoolVar(p *bool, name string, value bool, usage string) {
	add(name, newBoolValue(value, p), usage)
}

// Bool defines a bool flag with specified name, default value, and usage string.
// The return value is the address of a bool variable that stores the value of the flag.
func Bool(name string, value bool, usage string) *bool {
	p := new(bool);
	BoolVar(p, name, value, usage);
	return p;
}

// IntVar defines an int flag with specified name, default value, and usage string.
// The argument p points to an int variable in which to store the value of the flag.
func IntVar(p *int, name string, value int, usage string) {
	add(name, newIntValue(value, p), usage)
}

// Int defines an int flag with specified name, default value, and usage string.
// The return value is the address of an int variable that stores the value of the flag.
func Int(name string, value int, usage string) *int {
	p := new(int);
	IntVar(p, name, value, usage);
	return p;
}

// Int64Var defines an int64 flag with specified name, default value, and usage string.
// The argument p points to an int64 variable in which to store the value of the flag.
func Int64Var(p *int64, name string, value int64, usage string) {
	add(name, newInt64Value(value, p), usage)
}

// Int64 defines an int64 flag with specified name, default value, and usage string.
// The return value is the address of an int64 variable that stores the value of the flag.
func Int64(name string, value int64, usage string) *int64 {
	p := new(int64);
	Int64Var(p, name, value, usage);
	return p;
}

// UintVar defines a uint flag with specified name, default value, and usage string.
// The argument p points to a uint variable in which to store the value of the flag.
func UintVar(p *uint, name string, value uint, usage string) {
	add(name, newUintValue(value, p), usage)
}

// Uint defines a uint flag with specified name, default value, and usage string.
// The return value is the address of a uint variable that stores the value of the flag.
func Uint(name string, value uint, usage string) *uint {
	p := new(uint);
	UintVar(p, name, value, usage);
	return p;
}

// Uint64Var defines a uint64 flag with specified name, default value, and usage string.
// The argument p points to a uint64 variable in which to store the value of the flag.
func Uint64Var(p *uint64, name string, value uint64, usage string) {
	add(name, newUint64Value(value, p), usage)
}

// Uint64 defines a uint64 flag with specified name, default value, and usage string.
// The return value is the address of a uint64 variable that stores the value of the flag.
func Uint64(name string, value uint64, usage string) *uint64 {
	p := new(uint64);
	Uint64Var(p, name, value, usage);
	return p;
}

// StringVar defines a string flag with specified name, default value, and usage string.
// The argument p points to a string variable in which to store the value of the flag.
func StringVar(p *string, name, value string, usage string) {
	add(name, newStringValue(value, p), usage)
}

// String defines a string flag with specified name, default value, and usage string.
// The return value is the address of a string variable that stores the value of the flag.
func String(name, value string, usage string) *string {
	p := new(string);
	StringVar(p, name, value, usage);
	return p;
}

// FloatVar defines a float flag with specified name, default value, and usage string.
// The argument p points to a float variable in which to store the value of the flag.
func FloatVar(p *float, name string, value float, usage string) {
	add(name, newFloatValue(value, p), usage)
}

// Float defines a float flag with specified name, default value, and usage string.
// The return value is the address of a float variable that stores the value of the flag.
func Float(name string, value float, usage string) *float {
	p := new(float);
	FloatVar(p, name, value, usage);
	return p;
}

// Float64Var defines a float64 flag with specified name, default value, and usage string.
// The argument p points to a float64 variable in which to store the value of the flag.
func Float64Var(p *float64, name string, value float64, usage string) {
	add(name, newFloat64Value(value, p), usage)
}

// Float64 defines a float64 flag with specified name, default value, and usage string.
// The return value is the address of a float64 variable that stores the value of the flag.
func Float64(name string, value float64, usage string) *float64 {
	p := new(float64);
	Float64Var(p, name, value, usage);
	return p;
}


func (f *allFlags) parseOne(index int) (ok bool, next int) {
	s := os.Args[index];
	f.first_arg = index;	// until proven otherwise
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
			return false, index+1
		}
	}
	name := s[num_minuses : len(s)];
	if len(name) == 0 || name[0] == '-' || name[0] == '=' {
		fmt.Fprintln(os.Stderr, "bad flag syntax:", s);
		Usage();
		os.Exit(2);
	}

	// it's a flag. does it have an argument?
	has_value := false;
	value := "";
	for i := 1; i < len(name); i++ {	// equals cannot be first
		if name[i] == '=' {
			value = name[i+1 : len(name)];
			has_value = true;
			name = name[0:i];
			break;
		}
	}
	flag, alreadythere := flags.actual[name];
	if alreadythere {
		fmt.Fprintf(os.Stderr, "flag specified twice: -%s\n", name);
		Usage();
		os.Exit(2);
	}
	m := flags.formal;
	flag, alreadythere = m[name];	// BUG
	if !alreadythere {
		fmt.Fprintf(os.Stderr, "flag provided but not defined: -%s\n", name);
		Usage();
		os.Exit(2);
	}
	if f, ok := flag.Value.(*boolValue); ok {	// special case: doesn't need an arg
		if has_value {
			if !f.set(value) {
				fmt.Fprintf(os.Stderr, "invalid boolean value %t for flag: -%s\n", value, name);
				Usage();
				os.Exit(2);
			}
		} else {
			f.set("true")
		}
	} else {
		// It must have a value, which might be the next argument.
		if !has_value && index < len(os.Args)-1 {
			// value is the next arg
			has_value = true;
			index++;
			value = os.Args[index];
		}
		if !has_value {
			fmt.Fprintf(os.Stderr, "flag needs an argument: -%s\n", name);
			Usage();
			os.Exit(2);
		}
		ok = flag.Value.set(value);
		if !ok {
			fmt.Fprintf(os.Stderr, "invalid value %s for flag: -%s\n", value, name);
			Usage();
			os.Exit(2);
		}
	}
	flags.actual[name] = flag;
	return true, index+1;
}

// Parse parses the command-line flags.  Must be called after all flags are defined
// and before any are accessed by the program.
func Parse() {
	for i := 1; i < len(os.Args); {
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
