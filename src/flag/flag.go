// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//flag包实现了命令行参数的解析。
//
//Usage
//
//使用flag.String()、 Bool()、 Int()等函数定义flags。
//
//下例声明了一个整数flag： -n，解析结果保存在*int类型的指针nFlag中：
//
//	import "flag"
//	var nFlag = flag.Int("n", 1234, "help message for flag n")
//
//如果你喜欢，你可以使用Var系列函数将flag绑定到一个变量。
// 	var flagvar int
//	func init() {
//		flag.IntVar(&flagvar, "flagname", 1234, "help message for flagname")
//	}
//
//或者你可以创建满足Value接口（使用指针接收）的自定义flag，并且使用如下方式将其进行flag解析：
//	flag.Var(&flagVal, "name", "help message for flagname")
//对于这种flag，默认值就是该变量的初始值。
//
//在所有的flag都定义后，调用
//	flag.Parse()
//来解析命令行参数写入注册到已经定义的flag。
//
//解析之后，flag就可以直接使用了。如果你使用flag本身，它们是指针；如果你绑定到变量，它们是值。
//	fmt.Println("ip has value ", *ip)
//	fmt.Println("flagvar has value ", flagvar)
//
//解析之后，flag后面的参数可以从flag.Args()获取或用flag.Arg(i)单独获取。
//这些参数的索引是从0到flag.NArg()-1。
//
//Command line flag syntax
//
//允许以下格式：
//
//	-flag
//	-flag=x
//	-flag x  // 只有非bool类型的flag可以
//可以使用1个或者2个'-'号，效果是一样的。
//最后一种格式不能用于boolean类型的flag，原因是如果有文件名为0、false等，如下命令：
//	cmd -x *
//其含义会改变(*是Unix shell通配符)。你必须使用-flag=false格式来关闭一个boolean类型flag。
//
//Flag解析在第一个非flag参数（"-"是非flag参数）前或者在终止符"--"后停止
//
//整数类型flag接受1234、0664、0x1234，当然也可以是负数。
//Boolean类型flag可以是：
//	1, 0, t, f, T, F, true, false, TRUE, FALSE, True, False
//Duration类型的flag接受任何对time.ParseDuration有效的输入。
//
//命令行flag的默认集合是由顶层函数控制的。
//FlagSet类型允许程序员定义独立的flag集合，例如实现命令行界面下的子命令。
//FlagSet的方法类似于命令行flag集合的顶层函数
package flag

import (
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"
)

// 如果调用了-help或者-h标签，但是未定义此类标签，则返回ErrHelp错误。
var ErrHelp = errors.New("flag: help requested")

// errParse is returned by Set if a flag's value fails to parse, such as with an invalid integer for Int.
// It then gets wrapped through failf to provide more information.
var errParse = errors.New("parse error")

// errRange is returned by Set if a flag's value is out of range.
// It then gets wrapped through failf to provide more information.
var errRange = errors.New("value out of range")

func numError(err error) error {
	ne, ok := err.(*strconv.NumError)
	if !ok {
		return err
	}
	if ne.Err == strconv.ErrSyntax {
		return errParse
	}
	if ne.Err == strconv.ErrRange {
		return errRange
	}
	return err
}

// -- bool Value
type boolValue bool

func newBoolValue(val bool, p *bool) *boolValue {
	*p = val
	return (*boolValue)(p)
}

func (b *boolValue) Set(s string) error {
	v, err := strconv.ParseBool(s)
	if err != nil {
		err = errParse
	}
	*b = boolValue(v)
	return err
}

func (b *boolValue) Get() interface{} { return bool(*b) }

func (b *boolValue) String() string { return strconv.FormatBool(bool(*b)) }

func (b *boolValue) IsBoolFlag() bool { return true }

// optional interface to indicate boolean flags that can be
// supplied without "=value" text
type boolFlag interface {
	Value
	IsBoolFlag() bool
}

// -- int Value
type intValue int

func newIntValue(val int, p *int) *intValue {
	*p = val
	return (*intValue)(p)
}

func (i *intValue) Set(s string) error {
	v, err := strconv.ParseInt(s, 0, strconv.IntSize)
	if err != nil {
		err = numError(err)
	}
	*i = intValue(v)
	return err
}

func (i *intValue) Get() interface{} { return int(*i) }

func (i *intValue) String() string { return strconv.Itoa(int(*i)) }

// -- int64 Value
type int64Value int64

func newInt64Value(val int64, p *int64) *int64Value {
	*p = val
	return (*int64Value)(p)
}

func (i *int64Value) Set(s string) error {
	v, err := strconv.ParseInt(s, 0, 64)
	if err != nil {
		err = numError(err)
	}
	*i = int64Value(v)
	return err
}

func (i *int64Value) Get() interface{} { return int64(*i) }

func (i *int64Value) String() string { return strconv.FormatInt(int64(*i), 10) }

// -- uint Value
type uintValue uint

func newUintValue(val uint, p *uint) *uintValue {
	*p = val
	return (*uintValue)(p)
}

func (i *uintValue) Set(s string) error {
	v, err := strconv.ParseUint(s, 0, strconv.IntSize)
	if err != nil {
		err = numError(err)
	}
	*i = uintValue(v)
	return err
}

func (i *uintValue) Get() interface{} { return uint(*i) }

func (i *uintValue) String() string { return strconv.FormatUint(uint64(*i), 10) }

// -- uint64 Value
type uint64Value uint64

func newUint64Value(val uint64, p *uint64) *uint64Value {
	*p = val
	return (*uint64Value)(p)
}

func (i *uint64Value) Set(s string) error {
	v, err := strconv.ParseUint(s, 0, 64)
	if err != nil {
		err = numError(err)
	}
	*i = uint64Value(v)
	return err
}

func (i *uint64Value) Get() interface{} { return uint64(*i) }

func (i *uint64Value) String() string { return strconv.FormatUint(uint64(*i), 10) }

// -- string Value
type stringValue string

func newStringValue(val string, p *string) *stringValue {
	*p = val
	return (*stringValue)(p)
}

func (s *stringValue) Set(val string) error {
	*s = stringValue(val)
	return nil
}

func (s *stringValue) Get() interface{} { return string(*s) }

func (s *stringValue) String() string { return string(*s) }

// -- float64 Value
type float64Value float64

func newFloat64Value(val float64, p *float64) *float64Value {
	*p = val
	return (*float64Value)(p)
}

func (f *float64Value) Set(s string) error {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		err = numError(err)
	}
	*f = float64Value(v)
	return err
}

func (f *float64Value) Get() interface{} { return float64(*f) }

func (f *float64Value) String() string { return strconv.FormatFloat(float64(*f), 'g', -1, 64) }

// -- time.Duration Value
type durationValue time.Duration

func newDurationValue(val time.Duration, p *time.Duration) *durationValue {
	*p = val
	return (*durationValue)(p)
}

func (d *durationValue) Set(s string) error {
	v, err := time.ParseDuration(s)
	if err != nil {
		err = errParse
	}
	*d = durationValue(v)
	return err
}

func (d *durationValue) Get() interface{} { return time.Duration(*d) }

func (d *durationValue) String() string { return (*time.Duration)(d).String() }

// Value接口用于将动态的值保存在一个flag里（默认值被表示为一个字符串）。
//
// 如果Value接口具有IsBoolFlag()方法，且返回真，命令行解析会将-name等价于-name=true，而不是使用下一个命令行参数
//
// 对于每个存在的flag，Set会按顺序调用一次。
// flag包可以使用零值接收器(例如nil指针)调用String方法。
type Value interface {
	String() string
	Set(string) error
}

// Getter接口用于取回Value接口的内容。
// Getter是Go1之后添加的，并且出于兼容性考虑，它包装了Value接口，而不是作为Value接口的一部分。
// 本包中所有满足Value接口的类型都满足Getter接口
type Getter interface {
	Value
	Get() interface{}
}

// ErrorHandling定义了如何处理flag解析错误
type ErrorHandling int

// 如果解析失败，这些常量会作为FlagSet.Parse的行为描述
const (
	ContinueOnError ErrorHandling = iota // Return a descriptive error.
	ExitOnError                          // Call os.Exit(2) or for -h/-help Exit(0).
	PanicOnError                         // Call panic with a descriptive error.
)

// FlagSet代表一个已定义的flag结合。FlagSet的零值没有名称，并且采用ContinueOnError的错误处理策略。
//
// 在同一个FlagSet中flag名称必须是唯一的，尝试定义一个已存在的flag会导致panic。
type FlagSet struct {
	// 当解析flag发生错误时Usage函数将会被调用。
	// 该字段是一个函数（而非方法），以便修改为自定义的错误处理函数。
	// 当调用Usage后会发生什么取决于ErrorHandling设置；对于命令行，默认的处理策略是ExitOnError(退出程序)
	Usage func()

	name          string
	parsed        bool
	actual        map[string]*Flag
	formal        map[string]*Flag
	args          []string // arguments after flags
	errorHandling ErrorHandling
	output        io.Writer // nil means stderr; use Output() accessor
}

// Flag代表了一个flag的状态
type Flag struct {
	Name     string // flag在命令行中的名字
	Usage    string // 帮助信息
	Value    Value  // 设置的值
	DefValue string // 默认值（文本格式）；用于Usage
}

// sortFlags returns the flags as a slice in lexicographical sorted order.
func sortFlags(flags map[string]*Flag) []*Flag {
	result := make([]*Flag, len(flags))
	i := 0
	for _, f := range flags {
		result[i] = f
		i++
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i].Name < result[j].Name
	})
	return result
}

// Output返回使用信息和错误信息的目标，如果为设置Output或者将其设置为nil，则返回os.Stderr。
func (f *FlagSet) Output() io.Writer {
	if f.output == nil {
		return os.Stderr
	}
	return f.output
}

// Name返回FlagSet的名称。
func (f *FlagSet) Name() string {
	return f.name
}

// ErrorHandling返回FlagSet的错误处理行为。
func (f *FlagSet) ErrorHandling() ErrorHandling {
	return f.errorHandling
}

// SetOutput用于设置错误信息和使用信息的目标。
// 如果Output为nil，则使用os.Stderr。
func (f *FlagSet) SetOutput(output io.Writer) {
	f.output = output
}

// VisitAll按照字典顺序访问flag，对每个flag调用fn方法。
// 它会访问所有flag，包括没有设置的flag。
func (f *FlagSet) VisitAll(fn func(*Flag)) {
	for _, flag := range sortFlags(f.formal) {
		fn(flag)
	}
}

// VisitAll按照字典顺序访问命令行flag，对每个flag调用fn方法。
// 它会访问所有flag，包括没有设置的flag。
func VisitAll(fn func(*Flag)) {
	CommandLine.VisitAll(fn)
}

// VisitAll按照字典顺序访问flag，对每个flag调用fn方法。
// 它只访问设置的flag。
func (f *FlagSet) Visit(fn func(*Flag)) {
	for _, flag := range sortFlags(f.actual) {
		fn(flag)
	}
}

// VisitAll按照字典顺序访问命令行flag，对每个flag调用fn方法。
// 它只访问设置的flag。
func Visit(fn func(*Flag)) {
	CommandLine.Visit(fn)
}

// Lookup返回指定名称的Flag结构，如果没有找到返回nil。
func (f *FlagSet) Lookup(name string) *Flag {
	return f.formal[name]
}

// Lookup返回指定命令行flag名称的Flag结构，如果没有找到返回nil。
func Lookup(name string) *Flag {
	return CommandLine.formal[name]
}

// Set用于设置已注册的flag的值。
func (f *FlagSet) Set(name, value string) error {
	flag, ok := f.formal[name]
	if !ok {
		return fmt.Errorf("no such flag -%v", name)
	}
	err := flag.Value.Set(value)
	if err != nil {
		return err
	}
	if f.actual == nil {
		f.actual = make(map[string]*Flag)
	}
	f.actual[name] = flag
	return nil
}

// Set用于设置已注册的命令行flag的值。
func Set(name, value string) error {
	return CommandLine.Set(name, value)
}

// isZeroValue determines whether the string represents the zero
// value for a flag.
func isZeroValue(flag *Flag, value string) bool {
	// Build a zero value of the flag's Value type, and see if the
	// result of calling its String method equals the value passed in.
	// This works unless the Value type is itself an interface type.
	typ := reflect.TypeOf(flag.Value)
	var z reflect.Value
	if typ.Kind() == reflect.Ptr {
		z = reflect.New(typ.Elem())
	} else {
		z = reflect.Zero(typ)
	}
	return value == z.Interface().(Value).String()
}

// UnquoteUsage从使用信息中提起一个反引号标记的字符串并返回它和使用信息。
// 给定“要显示的名称”，它将返回（“名称”，“要显示的名称”）。
// 如果没有包含反引号，则该名称对应flag的值类型，如果是boolean类型，则为空字符串。
func UnquoteUsage(flag *Flag) (name string, usage string) {
	// 寻找一个反引号标记的名称，但不要使用strings包
	usage = flag.Usage
	for i := 0; i < len(usage); i++ {
		if usage[i] == '`' {
			for j := i + 1; j < len(usage); j++ {
				if usage[j] == '`' {
					name = usage[i+1 : j]
					usage = usage[:i] + name + usage[j+1:]
					return name, usage
				}
			}
			break // Only one back quote; use type name.
		}
	}
	// No explicit name, so use type if we can find one.
	name = "value"
	switch flag.Value.(type) {
	case boolFlag:
		name = ""
	case *durationValue:
		name = "duration"
	case *float64Value:
		name = "float"
	case *intValue, *int64Value:
		name = "int"
	case *stringValue:
		name = "string"
	case *uintValue, *uint64Value:
		name = "uint"
	}
	return
}

// PrintDefaults会打印所有集合中已经注册好的的默认值。除非另行配置，默认打印到标准错误输出中。
// 查看全局功能PrintDefaults文档获取更多信息。
func (f *FlagSet) PrintDefaults() {
	f.VisitAll(func(flag *Flag) {
		s := fmt.Sprintf("  -%s", flag.Name) // Two spaces before -; see next two comments.
		name, usage := UnquoteUsage(flag)
		if len(name) > 0 {
			s += " " + name
		}
		// Boolean flags of one ASCII letter are so common we
		// treat them specially, putting their usage on the same line.
		if len(s) <= 4 { // space, space, '-', 'x'.
			s += "\t"
		} else {
			// Four spaces before the tab triggers good alignment
			// for both 4- and 8-space tab stops.
			s += "\n    \t"
		}
		s += strings.ReplaceAll(usage, "\n", "\n    \t")

		if !isZeroValue(flag, flag.DefValue) {
			if _, ok := flag.Value.(*stringValue); ok {
				// put quotes on the value
				s += fmt.Sprintf(" (default %q)", flag.DefValue)
			} else {
				s += fmt.Sprintf(" (default %v)", flag.DefValue)
			}
		}
		fmt.Fprint(f.Output(), s, "\n")
	})
}

// PrintDefaults会打印所有集合中已经注册好的的默认值。除非另行配置，默认打印到标准错误输出中。
// 对于整形flag x，默认输出形式为：
//  -x int
//      usage-message-for-x (default 7)
// 通常使用信息都会显示在单独的一行，但是对于bool类型的flag，如果flag名称为一个字节，则使用信息在统一行。
// 如果类型的默认值是零值，则省略括号默认值。
// 可以通过在flag的使用信息中添加一个用反引号引起来的名称来更改列出的类型（此处为int）。
// 使用信息中第一个反引号引起来的名称会别视为在消息中显示的参数名称，并在显示时删除反引号。
// 例如，给定：
//  flag.String("I", "", "search `directory` for include files")
// 输入将是:
//  -I directory
//      search directory for include files.
//
// 若要改变消息的目标，请调用CommandLine.SetOutput
func PrintDefaults() {
	CommandLine.PrintDefaults()
}

// defaultUsage is the default function to print a usage message.
func (f *FlagSet) defaultUsage() {
	if f.name == "" {
		fmt.Fprintf(f.Output(), "Usage:\n")
	} else {
		fmt.Fprintf(f.Output(), "Usage of %s:\n", f.name)
	}
	f.PrintDefaults()
}

// 注意：Usage不只是defaultUsage（CommandLine）
// 因为它用作示例（godoc flag Usage）关于如何编写自定义的用法函数

// Usage会打印一条使用说明信息（记录所有已经定义的命令行flag）到CommandLine输出，默认是os.Stderr。
// 当解析flag发生错误时它会被调用。
// 该函数是一个变量，它可以更改为指向自定义函数。
// 默认情况下，它会打印一个简单的标头同时调用PrintDefaults函数；有关输出格式和如何控制它的详细信息请参考PrintDefaults函数的文档说明。
// 自定义的用法函数可以选择退出程序；默认情况下都会退出，因为命令行的错误处理策略设置为ExitOnError。
var Usage = func() {
	fmt.Fprintf(CommandLine.Output(), "Usage of %s:\n", os.Args[0])
	PrintDefaults()
}

// NFlag返回已经注册的flag的数量。
func (f *FlagSet) NFlag() int { return len(f.actual) }

// NFlag返回已经注册的命令行flag的数量。
func NFlag() int { return len(CommandLine.actual) }

// Arg返回第i个参数。Arg(0)是flag被解析后的第一个参数。如果请求的元素不存在，Arg返回空字符串。
func (f *FlagSet) Arg(i int) string {
	if i < 0 || i >= len(f.args) {
		return ""
	}
	return f.args[i]
}

// Arg返回第i个命令行参数。Arg(0)是flag被解析后的第一个参数。如果请求的元素不存在，Arg返回空字符串。
func Arg(i int) string {
	return CommandLine.Arg(i)
}

// NArg返回解析flag参数后剩余的参数数量。
func (f *FlagSet) NArg() int { return len(f.args) }

// NArg返回解析flag参数后剩余的参数数量。
func NArg() int { return len(CommandLine.args) }

//返回解析之后剩下的非flag参数。
func (f *FlagSet) Args() []string { return f.args }

//返回解析之后剩下的非flag命令行参数。
func Args() []string { return CommandLine.args }

// BoolVar用指定的名称、默认值、使用信息注册一个bool类型flag，并将flag的值保存到p指向的变量。
func (f *FlagSet) BoolVar(p *bool, name string, value bool, usage string) {
	f.Var(newBoolValue(value, p), name, usage)
}

// BoolVar用指定的名称、默认值、使用信息注册一个bool类型flag，并将flag的值保存到p指向的变量。
func BoolVar(p *bool, name string, value bool, usage string) {
	CommandLine.Var(newBoolValue(value, p), name, usage)
}

// Bool用指定的名称、默认值、使用信息注册一个bool类型flag。返回一个保存了该flag的值的指针。
func (f *FlagSet) Bool(name string, value bool, usage string) *bool {
	p := new(bool)
	f.BoolVar(p, name, value, usage)
	return p
}

// Bool用指定的名称、默认值、使用信息注册一个bool类型flag。返回一个保存了该flag的值的指针。
func Bool(name string, value bool, usage string) *bool {
	return CommandLine.Bool(name, value, usage)
}

// IntVar用指定的名称、默认值、使用信息注册一个int类型flag，并将flag的值保存到p指向的变量。
func (f *FlagSet) IntVar(p *int, name string, value int, usage string) {
	f.Var(newIntValue(value, p), name, usage)
}

// IntVar用指定的名称、默认值、使用信息注册一个int类型flag，并将flag的值保存到p指向的变量。
func IntVar(p *int, name string, value int, usage string) {
	CommandLine.Var(newIntValue(value, p), name, usage)
}

// Int用指定的名称、默认值、使用信息注册一个int类型flag。返回一个保存了该flag的值的指针。
func (f *FlagSet) Int(name string, value int, usage string) *int {
	p := new(int)
	f.IntVar(p, name, value, usage)
	return p
}

// Int用指定的名称、默认值、使用信息注册一个int类型flag。返回一个保存了该flag的值的指针。
func Int(name string, value int, usage string) *int {
	return CommandLine.Int(name, value, usage)
}

// Int64Var用指定的名称、默认值、使用信息注册一个int64类型flag，并将flag的值保存到p指向的变量。
func (f *FlagSet) Int64Var(p *int64, name string, value int64, usage string) {
	f.Var(newInt64Value(value, p), name, usage)
}

// Int64Var用指定的名称、默认值、使用信息注册一个int64类型flag，并将flag的值保存到p指向的变量。
func Int64Var(p *int64, name string, value int64, usage string) {
	CommandLine.Var(newInt64Value(value, p), name, usage)
}

// Int64用指定的名称、默认值、使用信息注册一个int64类型flag。返回一个保存了该flag的值的指针。
func (f *FlagSet) Int64(name string, value int64, usage string) *int64 {
	p := new(int64)
	f.Int64Var(p, name, value, usage)
	return p
}

// Int64用指定的名称、默认值、使用信息注册一个int64类型flag。返回一个保存了该flag的值的指针。
func Int64(name string, value int64, usage string) *int64 {
	return CommandLine.Int64(name, value, usage)
}

// UintVar用指定的名称、默认值、使用信息注册一个uint类型flag，并将flag的值保存到p指向的变量。
func (f *FlagSet) UintVar(p *uint, name string, value uint, usage string) {
	f.Var(newUintValue(value, p), name, usage)
}

// UintVar用指定的名称、默认值、使用信息注册一个uint类型flag，并将flag的值保存到p指向的变量。
func UintVar(p *uint, name string, value uint, usage string) {
	CommandLine.Var(newUintValue(value, p), name, usage)
}

// Uint用指定的名称、默认值、使用信息注册一个uint类型flag。返回一个保存了该flag的值的指针。
func (f *FlagSet) Uint(name string, value uint, usage string) *uint {
	p := new(uint)
	f.UintVar(p, name, value, usage)
	return p
}

// Uint用指定的名称、默认值、使用信息注册一个uint类型flag。返回一个保存了该flag的值的指针。
func Uint(name string, value uint, usage string) *uint {
	return CommandLine.Uint(name, value, usage)
}

// Uint64Var用指定的名称、默认值、使用信息注册一个uint64类型flag，并将flag的值保存到p指向的变量。
func (f *FlagSet) Uint64Var(p *uint64, name string, value uint64, usage string) {
	f.Var(newUint64Value(value, p), name, usage)
}

// Uint64Var用指定的名称、默认值、使用信息注册一个uint64类型flag，并将flag的值保存到p指向的变量。
func Uint64Var(p *uint64, name string, value uint64, usage string) {
	CommandLine.Var(newUint64Value(value, p), name, usage)
}

// Uint64用指定的名称、默认值、使用信息注册一个uint64类型flag。返回一个保存了该flag的值的指针。
func (f *FlagSet) Uint64(name string, value uint64, usage string) *uint64 {
	p := new(uint64)
	f.Uint64Var(p, name, value, usage)
	return p
}

// Uint64用指定的名称、默认值、使用信息注册一个uint64类型flag。返回一个保存了该flag的值的指针。
func Uint64(name string, value uint64, usage string) *uint64 {
	return CommandLine.Uint64(name, value, usage)
}

// StringVar用指定的名称、默认值、使用信息注册一个string类型flag，并将flag的值保存到p指向的变量。
func (f *FlagSet) StringVar(p *string, name string, value string, usage string) {
	f.Var(newStringValue(value, p), name, usage)
}

// StringVar用指定的名称、默认值、使用信息注册一个string类型flag，并将flag的值保存到p指向的变量。
func StringVar(p *string, name string, value string, usage string) {
	CommandLine.Var(newStringValue(value, p), name, usage)
}

// String用指定的名称、默认值、使用信息注册一个string类型flag。返回一个保存了该flag的值的指针。
func (f *FlagSet) String(name string, value string, usage string) *string {
	p := new(string)
	f.StringVar(p, name, value, usage)
	return p
}

// String用指定的名称、默认值、使用信息注册一个string类型flag。返回一个保存了该flag的值的指针。
func String(name string, value string, usage string) *string {
	return CommandLine.String(name, value, usage)
}

// Float64Var用指定的名称、默认值、使用信息注册一个float64类型flag，并将flag的值保存到p指向的变量。
func (f *FlagSet) Float64Var(p *float64, name string, value float64, usage string) {
	f.Var(newFloat64Value(value, p), name, usage)
}

// Float64Var用指定的名称、默认值、使用信息注册一个float64类型flag，并将flag的值保存到p指向的变量。
func Float64Var(p *float64, name string, value float64, usage string) {
	CommandLine.Var(newFloat64Value(value, p), name, usage)
}

// Float64用指定的名称、默认值、使用信息注册一个float64类型flag。返回一个保存了该flag的值的指针。
func (f *FlagSet) Float64(name string, value float64, usage string) *float64 {
	p := new(float64)
	f.Float64Var(p, name, value, usage)
	return p
}

// Float64用指定的名称、默认值、使用信息注册一个float64类型flag。返回一个保存了该flag的值的指针。
func Float64(name string, value float64, usage string) *float64 {
	return CommandLine.Float64(name, value, usage)
}

// DurationVar用指定的名称、默认值、使用信息注册一个time.Duration类型flag，并将flag的值保存到p指向的变量。
func (f *FlagSet) DurationVar(p *time.Duration, name string, value time.Duration, usage string) {
	f.Var(newDurationValue(value, p), name, usage)
}

// DurationVar用指定的名称、默认值、使用信息注册一个time.Duration类型flag，并将flag的值保存到p指向的变量。
func DurationVar(p *time.Duration, name string, value time.Duration, usage string) {
	CommandLine.Var(newDurationValue(value, p), name, usage)
}

// Duration用指定的名称、默认值、使用信息注册一个time.Duration类型flag。返回一个保存了该flag的值的指针。
func (f *FlagSet) Duration(name string, value time.Duration, usage string) *time.Duration {
	p := new(time.Duration)
	f.DurationVar(p, name, value, usage)
	return p
}

// Duration用指定的名称、默认值、使用信息注册一个time.Duration类型flag。返回一个保存了该flag的值的指针。
func Duration(name string, value time.Duration, usage string) *time.Duration {
	return CommandLine.Duration(name, value, usage)
}

// Var方法使用指定的名字、使用信息注册一个flag。
// 该flag的类型和值由第一个参数表示，该参数应实现了Value接口。
// 例如，用户可以创建一个flag，可以用Value接口的Set方法将逗号分隔的字符串转化为字符串切片。
func (f *FlagSet) Var(value Value, name string, usage string) {
	// Remember the default value as a string; it won't change.
	flag := &Flag{name, usage, value, value.String()}
	_, alreadythere := f.formal[name]
	if alreadythere {
		var msg string
		if f.name == "" {
			msg = fmt.Sprintf("flag redefined: %s", name)
		} else {
			msg = fmt.Sprintf("%s flag redefined: %s", f.name, name)
		}
		fmt.Fprintln(f.Output(), msg)
		panic(msg) // Happens only if flags are declared with identical names
	}
	if f.formal == nil {
		f.formal = make(map[string]*Flag)
	}
	f.formal[name] = flag
}

// Var方法使用指定的名字、使用信息注册一个flag。
// 该flag的类型和值由第一个参数表示，该参数应实现了Value接口。
// 例如，用户可以创建一个flag，可以用Value接口的Set方法将逗号分隔的字符串转化为字符串切片。
func Var(value Value, name string, usage string) {
	CommandLine.Var(value, name, usage)
}

// failf将格式化的错误和实用信息打印到标准错误并返回。
func (f *FlagSet) failf(format string, a ...interface{}) error {
	err := fmt.Errorf(format, a...)
	fmt.Fprintln(f.Output(), err)
	f.usage()
	return err
}

// usage calls the Usage method for the flag set if one is specified,
// or the appropriate default usage function otherwise.
func (f *FlagSet) usage() {
	if f.Usage == nil {
		f.defaultUsage()
	} else {
		f.Usage()
	}
}

// parseOne parses one flag. It reports whether a flag was seen.
func (f *FlagSet) parseOne() (bool, error) {
	if len(f.args) == 0 {
		return false, nil
	}
	s := f.args[0]
	if len(s) < 2 || s[0] != '-' {
		return false, nil
	}
	numMinuses := 1
	if s[1] == '-' {
		numMinuses++
		if len(s) == 2 { // "--" terminates the flags
			f.args = f.args[1:]
			return false, nil
		}
	}
	name := s[numMinuses:]
	if len(name) == 0 || name[0] == '-' || name[0] == '=' {
		return false, f.failf("bad flag syntax: %s", s)
	}

	// it's a flag. does it have an argument?
	f.args = f.args[1:]
	hasValue := false
	value := ""
	for i := 1; i < len(name); i++ { // equals cannot be first
		if name[i] == '=' {
			value = name[i+1:]
			hasValue = true
			name = name[0:i]
			break
		}
	}
	m := f.formal
	flag, alreadythere := m[name] // BUG
	if !alreadythere {
		if name == "help" || name == "h" { // special case for nice help message.
			f.usage()
			return false, ErrHelp
		}
		return false, f.failf("flag provided but not defined: -%s", name)
	}

	if fv, ok := flag.Value.(boolFlag); ok && fv.IsBoolFlag() { // special case: doesn't need an arg
		if hasValue {
			if err := fv.Set(value); err != nil {
				return false, f.failf("invalid boolean value %q for -%s: %v", value, name, err)
			}
		} else {
			if err := fv.Set("true"); err != nil {
				return false, f.failf("invalid boolean flag %s: %v", name, err)
			}
		}
	} else {
		// It must have a value, which might be the next argument.
		if !hasValue && len(f.args) > 0 {
			// value is the next arg
			hasValue = true
			value, f.args = f.args[0], f.args[1:]
		}
		if !hasValue {
			return false, f.failf("flag needs an argument: -%s", name)
		}
		if err := flag.Value.Set(value); err != nil {
			return false, f.failf("invalid value %q for flag -%s: %v", value, name, err)
		}
	}
	if f.actual == nil {
		f.actual = make(map[string]*Flag)
	}
	f.actual[name] = flag
	return true, nil
}

// 从arguments中解析注册的flag。
// 必须在所有flag都注册好而未访问其值时执行。未注册却使用flag -help时，会返回ErrHelp。
func (f *FlagSet) Parse(arguments []string) error {
	f.parsed = true
	f.args = arguments
	for {
		seen, err := f.parseOne()
		if seen {
			continue
		}
		if err == nil {
			break
		}
		switch f.errorHandling {
		case ContinueOnError:
			return err
		case ExitOnError:
			if err == ErrHelp {
				os.Exit(0)
			}
			os.Exit(2)
		case PanicOnError:
			panic(err)
		}
	}
	return nil
}

// 返回是否f.Parse已经被调用过。
func (f *FlagSet) Parsed() bool {
	return f.parsed
}

// 从os.Args[1:]中解析注册的命令行flag。必须在所有flag都注册好而未访问其值时执行。
// 未注册却使用flag -help时，会返回ErrHelp。
func Parse() {
	// Ignore errors; CommandLine is set for ExitOnError.
	CommandLine.Parse(os.Args[1:])
}

// 返回是否f.Parse已经被调用过。
func Parsed() bool {
	return CommandLine.Parsed()
}

// CommandLine 是从os.Args解析的默认命令行flag集合。
//BoolVar、Arg等顶层函数是CommandLine方法的包装。
var CommandLine = NewFlagSet(os.Args[0], ExitOnError)

func init() {
	// Override generic FlagSet default Usage with call to global Usage.
	// Note: This is not CommandLine.Usage = Usage,
	// because we want any eventual call to use any updated value of Usage,
	// not the value it has when this line is run.
	CommandLine.Usage = commandLineUsage
}

func commandLineUsage() {
	Usage()
}

// NewFlagSet返回一个带有指定名称的新的空flag，以及一个错误处理属性。
// 如果名称不为空，则将其打印到默认的使用信息和错误信息中。
func NewFlagSet(name string, errorHandling ErrorHandling) *FlagSet {
	f := &FlagSet{
		name:          name,
		errorHandling: errorHandling,
	}
	f.Usage = f.defaultUsage
	return f
}

// Init设置flag集合的名字和错误处理属性。
// 默认情况下，FlagSet零值没有名字，采用ContinueOnError错误处理策略。
func (f *FlagSet) Init(name string, errorHandling ErrorHandling) {
	f.name = name
	f.errorHandling = errorHandling
}
