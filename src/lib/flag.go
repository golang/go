// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flag

/*
 * Flags
 *
 * Usage:
 *	1) Define flags using flag.String(), Bool(), or Int(). Int flag values have type int64. Example:
 *		import flag "flag"
 *		var i int64
 *		var fi *flag.Flag = flag.Int("flagname", 1234, &i, "help message for flagname")
 *	The pointer may be nil; if non-nil, it points to a cell of the appropriate type to store the
 *	flag's value.
 *
 *	2) After all flags are defined, call
 *		flag.Parse()
 *	to parse the command line into the defined flags.
 *
 *	3) Flags may then be used directly (getters are SVal, BVal, Ival) or through the associated
 *	cell, if set:
 *		print("fi has value ", fi.IVal(), "\n");
 *		print("i has value ", i, "\n");
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

import fmt "fmt"

//export Bool, Int, String
//export Arg, NArg
//export Parse
//export Flag.BVal BUG: variable exported but not defined: Flag.BVal
//export Flag.SVal BUG: variable exported but not defined: Flag.SVal
//export Flag

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
	BoolValue struct;
	IntValue struct;
	StringValue struct;
)

// -- Bool Value
type BoolValue struct {
	val bool;
	p *bool;
}

func NewBoolValue(val bool, p *bool) *BoolValue {
	if p != nil {
		*p = val
	}
	return &BoolValue{val, p}
}

func (b *BoolValue) AsBool() *BoolValue {
	return b
}

func (b *BoolValue) AsInt() *IntValue {
	return nil
}

func (b *BoolValue) AsString() *StringValue {
	return nil
}

func (b *BoolValue) IsBool() bool {
	return true
}

func (b *BoolValue) IsInt() bool {
	return false
}

func (b *BoolValue) IsString() bool {
	return false
}

func (b *BoolValue) ValidValue(str string) bool {
	i, ok := atob(str);
	return ok;
}

func (b *BoolValue) Set(val bool) {
	if b.p != nil {
		*b.p = val
	}
	b.val = val
}

func (b *BoolValue) Str() string {
	if b.val {
		return "true"
	}
	return "false"
}

// -- Int Value
type IntValue struct {
	val	int64;
	p	*int64;
}

func NewIntValue(val int64, p *int64) *IntValue {
	if p != nil {
		*p = val
	}
	return &IntValue{val, p}
}

func (i *IntValue) AsBool() *BoolValue {
	return nil
}

func (i *IntValue) AsInt() *IntValue {
	return i
}

func (i *IntValue) AsString() *StringValue{
	return nil
}

func (i *IntValue) IsBool() bool {
	return false
}

func (i *IntValue) IsInt() bool {
	return true
}

func (i *IntValue) IsString() bool {
	return false
}

func (i *IntValue) ValidValue(str string) bool {
	k, ok := atoi(str);
	return ok;
}

func (i *IntValue) Set(val int64) {
	if i.p != nil {
		*i.p = val
	}
	i.val = val
}

func (i *IntValue) Str() string {
	return fmt.New().d64(i.val).str()
}

// -- String Value
type StringValue struct {
	val	string;
	p	*string;
}

func NewStringValue(val string, p *string) *StringValue {
	if p != nil {
		*p = val
	}
	return &StringValue{val, p}
}

func (e *StringValue) AsBool() *BoolValue {
	return nil
}

func (e *StringValue) AsInt() *IntValue {
	return nil
}

func (s *StringValue) AsString() *StringValue{
	return s
}

func (s *StringValue) IsBool() bool {
	return false
}

func (s *StringValue) IsInt() bool {
	return false
}

func (s *StringValue) IsString() bool {
	return true
}

func (s *StringValue) ValidValue(str string) bool {
	return true
}

func (s *StringValue) Set(val string) {
	if s.p != nil {
		*s.p = val
	}
	s.val = val
}

func (s *StringValue) Str() string {
	return `"` + s.val + `"`
}

// -- Value interface
type Value interface {
	AsBool()	*BoolValue;
	AsInt()	*IntValue;
	AsString()	*StringValue;
	IsBool()	bool;
	IsInt()	bool;
	IsString()	bool;
	Str()		string;
	ValidValue(str string) bool;
}

// -- Flag structure (internal)
export type Flag struct {
	name	string;
	usage	string;
	value	Value;
	next	*Flag;  // BUG: remove when we can iterate over maps
}

type Flags struct {
	actual *map[string] *Flag;
	formal *map[string] *Flag;
	first_arg	int;
	flag_list	*Flag;  // BUG: remove when we can iterate over maps
}

// --Customer's value getters
func (f *Flag) BVal() bool {
	if !f.value.IsBool() {
		return false;
	}
	return f.value.AsBool().val;
}

func (f *Flag) IVal() int64 {
	if !f.value.IsInt() {
		return 0
	}
	return f.value.AsInt().val;
}

func (f *Flag) SVal() string {
	if !f.value.IsString() {
		return "???";
	}
	return f.value.AsString().val;
}

func New() *Flags {
	f := new(Flags);
	f.first_arg = 1;	// 0 is the program name, 1 is first arg
	f.actual = new(map[string] *Flag);
	f.formal = new(map[string] *Flag);
	return f;
}

var flags *Flags = New();

export func PrintDefaults() {
	// BUG: use map iteration when available
	for f := flags.flag_list; f != nil; f = f.next {
		print("  -", f.name, "=", f.value.Str(), ": ", f.usage, "\n");
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

func Add(name string, value Value, usage string) *Flag {
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
	f.next = flags.flag_list;  // BUG: remove when we can iterate over maps
	flags.flag_list = f;  // BUG: remove when we can iterate over maps
	return f;
}

export func Bool(name string, value bool, p *bool, usage string) *Flag {
	return Add(name, NewBoolValue(value, p), usage);
}

export func Int(name string, value int64, p *int64, usage string) *Flag {
	return Add(name, NewIntValue(value, p), usage);
}

export func String(name, value string, p *string, usage string) *Flag {
	return Add(name, NewStringValue(value, p), usage);
}

func (f *Flags) ParseOne(index int) (ok bool, next int)
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
	if !has_value && index < sys.argc()-1 && flag.value.ValidValue(sys.argv(index+1)) {
		// value is the next arg
		has_value = true;
		index++;
		value = sys.argv(index);
	}
	switch {
		case flag.value.IsBool():
			if has_value {
				k, ok := atob(value);
				if !ok {
					print("invalid boolean value ", value, " for flag: -", name, "\n");
					Usage();
				}
				flag.value.AsBool().Set(k)
			} else {
				flag.value.AsBool().Set(true)
			}
		case flag.value.IsInt():
			if !has_value {
				print("flag needs an argument: -", name, "\n");
				Usage();
			}
			k, ok := atoi(value);
			if !ok {
				print("invalid integer value ", value, " for flag: -", name, "\n");
				Usage();
			}
			flag.value.AsInt().Set(k);
		case flag.value.IsString():
			if !has_value {
				print("flag needs an argument: -", name, "\n");
				Usage();
			}
			flag.value.AsString().Set(value)
	}
	flags.actual[name] = flag;
	return true, index + 1
}

export func Parse() {
	for i := 1; i < sys.argc();  {
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
