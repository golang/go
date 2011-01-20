// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	The flag package implements command-line flag parsing.

	Usage:

	Define flags using flag.String(), Bool(), Int(), etc. Example:
		import "flag"
		var ip *int = flag.Int("flagname", 1234, "help message for flagname")
	If you like, you can bind the flag to a variable using the Var() functions.
		var flagvar int
		func init() {
			flag.IntVar(&flagvar, "flagname", 1234, "help message for flagname")
		}
	Or you can create custom flags that satisfy the Value interface (with
	pointer receivers) and couple them to flag parsing by
		flag.Var(&flagVal, "name", "help message for flagname")
	For such flags, the default value is just the initial value of the variable.

	After all flags are defined, call
		flag.Parse()
	to parse the command line into the defined flags.

	Flags may then be used directly. If you're using the flags themselves,
	they are all pointers; if you bind to variables, they're values.
		fmt.Println("ip has value ", *ip);
		fmt.Println("flagvar has value ", flagvar);

	After parsing, the arguments after the flag are available as the
	slice flag.Args() or individually as flag.Arg(i).
	The arguments are indexed from 0 up to flag.NArg().

	Command line flag syntax:
		-flag
		-flag=x
		-flag x  // non-boolean flags only
	One or two minus signs may be used; they are equivalent.
	The last form is not permitted for boolean flags because the
	meaning of the command
		cmd -x *
	will change if there is a file called 0, false, etc.  You must
	use the -flag=false form to turn off a boolean flag.

	Flag parsing stops just before the first non-flag argument
	("-" is a non-flag argument) or after the terminator "--".

	Integer flags accept 1234, 0664, 0x1234 and may be negative.
	Boolean flags may be 1, 0, t, f, true, false, TRUE, FALSE, True, False.

	It is safe to call flag.Parse multiple times, possibly after changing
	os.Args.  This makes it possible to implement command lines with
	subcommands that enable additional flags, as in:

		flag.Bool(...)  // global options
		flag.Parse()  // parse leading command
		subcmd := flag.Args(0)
		switch subcmd {
			// add per-subcommand options
		}
		os.Args = flag.Args()
		flag.Parse()
*/
package flag

import (
	"fmt"
	"os"
	"strconv"
)

// -- Bool Value
type boolValue bool

func newBoolValue(val bool, p *bool) *boolValue {
	*p = val
	return (*boolValue)(p)
}

func (b *boolValue) Set(s string) bool {
	v, err := strconv.Atob(s)
	*b = boolValue(v)
	return err == nil
}

func (b *boolValue) String() string { return fmt.Sprintf("%v", *b) }

// -- Int Value
type intValue int

func newIntValue(val int, p *int) *intValue {
	*p = val
	return (*intValue)(p)
}

func (i *intValue) Set(s string) bool {
	v, err := strconv.Atoi(s)
	*i = intValue(v)
	return err == nil
}

func (i *intValue) String() string { return fmt.Sprintf("%v", *i) }

// -- Int64 Value
type int64Value int64

func newInt64Value(val int64, p *int64) *int64Value {
	*p = val
	return (*int64Value)(p)
}

func (i *int64Value) Set(s string) bool {
	v, err := strconv.Atoi64(s)
	*i = int64Value(v)
	return err == nil
}

func (i *int64Value) String() string { return fmt.Sprintf("%v", *i) }

// -- Uint Value
type uintValue uint

func newUintValue(val uint, p *uint) *uintValue {
	*p = val
	return (*uintValue)(p)
}

func (i *uintValue) Set(s string) bool {
	v, err := strconv.Atoui(s)
	*i = uintValue(v)
	return err == nil
}

func (i *uintValue) String() string { return fmt.Sprintf("%v", *i) }

// -- uint64 Value
type uint64Value uint64

func newUint64Value(val uint64, p *uint64) *uint64Value {
	*p = val
	return (*uint64Value)(p)
}

func (i *uint64Value) Set(s string) bool {
	v, err := strconv.Atoui64(s)
	*i = uint64Value(v)
	return err == nil
}

func (i *uint64Value) String() string { return fmt.Sprintf("%v", *i) }

// -- string Value
type stringValue string

func newStringValue(val string, p *string) *stringValue {
	*p = val
	return (*stringValue)(p)
}

func (s *stringValue) Set(val string) bool {
	*s = stringValue(val)
	return true
}

func (s *stringValue) String() string { return fmt.Sprintf("%s", *s) }

// -- Float64 Value
type float64Value float64

func newFloat64Value(val float64, p *float64) *float64Value {
	*p = val
	return (*float64Value)(p)
}

func (f *float64Value) Set(s string) bool {
	v, err := strconv.Atof64(s)
	*f = float64Value(v)
	return err == nil
}

func (f *float64Value) String() string { return fmt.Sprintf("%v", *f) }

// Value is the interface to the dynamic value stored in a flag.
// (The default value is represented as a string.)
type Value interface {
	String() string
	Set(string) bool
}

// A Flag represents the state of a flag.
type Flag struct {
	Name     string // name as it appears on command line
	Usage    string // help message
	Value    Value  // value as set
	DefValue string // default value (as text); for usage message
}

type allFlags struct {
	actual map[string]*Flag
	formal map[string]*Flag
	args   []string // arguments after flags
}

var flags *allFlags

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
	return flags.formal[name]
}

// Set sets the value of the named flag.  It returns true if the set succeeded; false if
// there is no such flag defined.
func Set(name, value string) bool {
	f, ok := flags.formal[name]
	if !ok {
		return false
	}
	ok = f.Value.Set(value)
	if !ok {
		return false
	}
	flags.actual[name] = f
	return true
}

// PrintDefaults prints to standard error the default values of all defined flags.
func PrintDefaults() {
	VisitAll(func(f *Flag) {
		format := "  -%s=%s: %s\n"
		if _, ok := f.Value.(*stringValue); ok {
			// put quotes on the value
			format = "  -%s=%q: %s\n"
		}
		fmt.Fprintf(os.Stderr, format, f.Name, f.DefValue, f.Usage)
	})
}

// Usage prints to standard error a default usage message documenting all defined flags.
// The function is a variable that may be changed to point to a custom function.
var Usage = func() {
	fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
	PrintDefaults()
}

var panicOnError = false

func fail() {
	Usage()
	if panicOnError {
		panic("flag parse error")
	}
	os.Exit(2)
}

func NFlag() int { return len(flags.actual) }

// Arg returns the i'th command-line argument.  Arg(0) is the first remaining argument
// after flags have been processed.
func Arg(i int) string {
	if i < 0 || i >= len(flags.args) {
		return ""
	}
	return flags.args[i]
}

// NArg is the number of arguments remaining after flags have been processed.
func NArg() int { return len(flags.args) }

// Args returns the non-flag command-line arguments.
func Args() []string { return flags.args }

// BoolVar defines a bool flag with specified name, default value, and usage string.
// The argument p points to a bool variable in which to store the value of the flag.
func BoolVar(p *bool, name string, value bool, usage string) {
	Var(newBoolValue(value, p), name, usage)
}

// Bool defines a bool flag with specified name, default value, and usage string.
// The return value is the address of a bool variable that stores the value of the flag.
func Bool(name string, value bool, usage string) *bool {
	p := new(bool)
	BoolVar(p, name, value, usage)
	return p
}

// IntVar defines an int flag with specified name, default value, and usage string.
// The argument p points to an int variable in which to store the value of the flag.
func IntVar(p *int, name string, value int, usage string) {
	Var(newIntValue(value, p), name, usage)
}

// Int defines an int flag with specified name, default value, and usage string.
// The return value is the address of an int variable that stores the value of the flag.
func Int(name string, value int, usage string) *int {
	p := new(int)
	IntVar(p, name, value, usage)
	return p
}

// Int64Var defines an int64 flag with specified name, default value, and usage string.
// The argument p points to an int64 variable in which to store the value of the flag.
func Int64Var(p *int64, name string, value int64, usage string) {
	Var(newInt64Value(value, p), name, usage)
}

// Int64 defines an int64 flag with specified name, default value, and usage string.
// The return value is the address of an int64 variable that stores the value of the flag.
func Int64(name string, value int64, usage string) *int64 {
	p := new(int64)
	Int64Var(p, name, value, usage)
	return p
}

// UintVar defines a uint flag with specified name, default value, and usage string.
// The argument p points to a uint variable in which to store the value of the flag.
func UintVar(p *uint, name string, value uint, usage string) {
	Var(newUintValue(value, p), name, usage)
}

// Uint defines a uint flag with specified name, default value, and usage string.
// The return value is the address of a uint variable that stores the value of the flag.
func Uint(name string, value uint, usage string) *uint {
	p := new(uint)
	UintVar(p, name, value, usage)
	return p
}

// Uint64Var defines a uint64 flag with specified name, default value, and usage string.
// The argument p points to a uint64 variable in which to store the value of the flag.
func Uint64Var(p *uint64, name string, value uint64, usage string) {
	Var(newUint64Value(value, p), name, usage)
}

// Uint64 defines a uint64 flag with specified name, default value, and usage string.
// The return value is the address of a uint64 variable that stores the value of the flag.
func Uint64(name string, value uint64, usage string) *uint64 {
	p := new(uint64)
	Uint64Var(p, name, value, usage)
	return p
}

// StringVar defines a string flag with specified name, default value, and usage string.
// The argument p points to a string variable in which to store the value of the flag.
func StringVar(p *string, name, value string, usage string) {
	Var(newStringValue(value, p), name, usage)
}

// String defines a string flag with specified name, default value, and usage string.
// The return value is the address of a string variable that stores the value of the flag.
func String(name, value string, usage string) *string {
	p := new(string)
	StringVar(p, name, value, usage)
	return p
}

// Float64Var defines a float64 flag with specified name, default value, and usage string.
// The argument p points to a float64 variable in which to store the value of the flag.
func Float64Var(p *float64, name string, value float64, usage string) {
	Var(newFloat64Value(value, p), name, usage)
}

// Float64 defines a float64 flag with specified name, default value, and usage string.
// The return value is the address of a float64 variable that stores the value of the flag.
func Float64(name string, value float64, usage string) *float64 {
	p := new(float64)
	Float64Var(p, name, value, usage)
	return p
}

// Var defines a user-typed flag with specified name, default value, and usage string.
// The argument p points to a Value variable in which to store the value of the flag.
func Var(value Value, name string, usage string) {
	// Remember the default value as a string; it won't change.
	f := &Flag{name, usage, value, value.String()}
	_, alreadythere := flags.formal[name]
	if alreadythere {
		fmt.Fprintln(os.Stderr, "flag redefined:", name)
		panic("flag redefinition") // Happens only if flags are declared with identical names
	}
	flags.formal[name] = f
}


func (f *allFlags) parseOne() (ok bool) {
	if len(f.args) == 0 {
		return false
	}
	s := f.args[0]
	if len(s) == 0 || s[0] != '-' || len(s) == 1 {
		return false
	}
	num_minuses := 1
	if s[1] == '-' {
		num_minuses++
		if len(s) == 2 { // "--" terminates the flags
			f.args = f.args[1:]
			return false
		}
	}
	name := s[num_minuses:]
	if len(name) == 0 || name[0] == '-' || name[0] == '=' {
		fmt.Fprintln(os.Stderr, "bad flag syntax:", s)
		fail()
	}

	// it's a flag. does it have an argument?
	f.args = f.args[1:]
	has_value := false
	value := ""
	for i := 1; i < len(name); i++ { // equals cannot be first
		if name[i] == '=' {
			value = name[i+1:]
			has_value = true
			name = name[0:i]
			break
		}
	}
	m := flags.formal
	flag, alreadythere := m[name] // BUG
	if !alreadythere {
		fmt.Fprintf(os.Stderr, "flag provided but not defined: -%s\n", name)
		fail()
	}
	if fv, ok := flag.Value.(*boolValue); ok { // special case: doesn't need an arg
		if has_value {
			if !fv.Set(value) {
				fmt.Fprintf(os.Stderr, "invalid boolean value %q for flag: -%s\n", value, name)
				fail()
			}
		} else {
			fv.Set("true")
		}
	} else {
		// It must have a value, which might be the next argument.
		if !has_value && len(f.args) > 0 {
			// value is the next arg
			has_value = true
			value, f.args = f.args[0], f.args[1:]
		}
		if !has_value {
			fmt.Fprintf(os.Stderr, "flag needs an argument: -%s\n", name)
			fail()
		}
		ok = flag.Value.Set(value)
		if !ok {
			fmt.Fprintf(os.Stderr, "invalid value %q for flag: -%s\n", value, name)
			fail()
		}
	}
	flags.actual[name] = flag
	return true
}

// Parse parses the command-line flags.  Must be called after all flags are defined
// and before any are accessed by the program.
func Parse() {
	flags.args = os.Args[1:]
	for flags.parseOne() {
	}
}

func init() {
	flags = &allFlags{make(map[string]*Flag), make(map[string]*Flag), os.Args[1:]}
}
