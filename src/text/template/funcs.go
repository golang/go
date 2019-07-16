// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/url"
	"reflect"
	"strings"
	"unicode"
	"unicode/utf8"
)

// FuncMap is the type of the map defining the mapping from names to functions.
// Each function must have either a single return value, or two return values of
// which the second has type error. In that case, if the second (error)
// return value evaluates to non-nil during execution, execution terminates and
// Execute returns that error.
//
// When template execution invokes a function with an argument list, that list
// must be assignable to the function's parameter types. Functions meant to
// apply to arguments of arbitrary type can use parameters of type interface{} or
// of type reflect.Value. Similarly, functions meant to return a result of arbitrary
// type can return interface{} or reflect.Value.
type FuncMap map[string]interface{}

var builtins = FuncMap{
	"and":      and,
	"call":     call,
	"html":     HTMLEscaper,
	"index":    index,
	"slice":    slice,
	"js":       JSEscaper,
	"len":      length,
	"not":      not,
	"or":       or,
	"print":    fmt.Sprint,
	"printf":   fmt.Sprintf,
	"println":  fmt.Sprintln,
	"urlquery": URLQueryEscaper,

	// Comparisons
	"eq": eq, // ==
	"ge": ge, // >=
	"gt": gt, // >
	"le": le, // <=
	"lt": lt, // <
	"ne": ne, // !=
}

var builtinFuncs = createValueFuncs(builtins)

// createValueFuncs turns a FuncMap into a map[string]reflect.Value
func createValueFuncs(funcMap FuncMap) map[string]reflect.Value {
	m := make(map[string]reflect.Value)
	addValueFuncs(m, funcMap)
	return m
}

// addValueFuncs adds to values the functions in funcs, converting them to reflect.Values.
func addValueFuncs(out map[string]reflect.Value, in FuncMap) {
	for name, fn := range in {
		if !goodName(name) {
			panic(fmt.Errorf("function name %q is not a valid identifier", name))
		}
		v := reflect.ValueOf(fn)
		if v.Kind() != reflect.Func {
			panic("value for " + name + " not a function")
		}
		if !goodFunc(v.Type()) {
			panic(fmt.Errorf("can't install method/function %q with %d results", name, v.Type().NumOut()))
		}
		out[name] = v
	}
}

// addFuncs adds to values the functions in funcs. It does no checking of the input -
// call addValueFuncs first.
func addFuncs(out, in FuncMap) {
	for name, fn := range in {
		out[name] = fn
	}
}

// goodFunc reports whether the function or method has the right result signature.
func goodFunc(typ reflect.Type) bool {
	// We allow functions with 1 result or 2 results where the second is an error.
	switch {
	case typ.NumOut() == 1:
		return true
	case typ.NumOut() == 2 && typ.Out(1) == errorType:
		return true
	}
	return false
}

// goodName reports whether the function name is a valid identifier.
func goodName(name string) bool {
	if name == "" {
		return false
	}
	for i, r := range name {
		switch {
		case r == '_':
		case i == 0 && !unicode.IsLetter(r):
			return false
		case !unicode.IsLetter(r) && !unicode.IsDigit(r):
			return false
		}
	}
	return true
}

// findFunction looks for a function in the template, and global map.
func findFunction(name string, tmpl *Template) (reflect.Value, bool) {
	if tmpl != nil && tmpl.common != nil {
		tmpl.muFuncs.RLock()
		defer tmpl.muFuncs.RUnlock()
		if fn := tmpl.execFuncs[name]; fn.IsValid() {
			return fn, true
		}
	}
	if fn := builtinFuncs[name]; fn.IsValid() {
		return fn, true
	}
	return reflect.Value{}, false
}

// prepareArg checks if value can be used as an argument of type argType, and
// converts an invalid value to appropriate zero if possible.
func prepareArg(value reflect.Value, argType reflect.Type) (reflect.Value, error) {
	if !value.IsValid() {
		if !canBeNil(argType) {
			return reflect.Value{}, fmt.Errorf("value is nil; should be of type %s", argType)
		}
		value = reflect.Zero(argType)
	}
	if value.Type().AssignableTo(argType) {
		return value, nil
	}
	if intLike(value.Kind()) && intLike(argType.Kind()) && value.Type().ConvertibleTo(argType) {
		value = value.Convert(argType)
		return value, nil
	}
	return reflect.Value{}, fmt.Errorf("value has type %s; should be %s", value.Type(), argType)
}

func intLike(typ reflect.Kind) bool {
	switch typ {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return true
	}
	return false
}

// indexArg checks if a reflect.Value can be used as an index, and converts it to int if possible.
func indexArg(index reflect.Value, cap int) (int, error) {
	var x int64
	switch index.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		x = index.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		x = int64(index.Uint())
	case reflect.Invalid:
		return 0, fmt.Errorf("cannot index slice/array with nil")
	default:
		return 0, fmt.Errorf("cannot index slice/array with type %s", index.Type())
	}
	if x < 0 || int(x) < 0 || int(x) > cap {
		return 0, fmt.Errorf("index out of range: %d", x)
	}
	return int(x), nil
}

// Indexing.

// index returns the result of indexing its first argument by the following
// arguments. Thus "index x 1 2 3" is, in Go syntax, x[1][2][3]. Each
// indexed item must be a map, slice, or array.
func index(item reflect.Value, indexes ...reflect.Value) (reflect.Value, error) {
	v := indirectInterface(item)
	if !v.IsValid() {
		return reflect.Value{}, fmt.Errorf("index of untyped nil")
	}
	for _, i := range indexes {
		index := indirectInterface(i)
		var isNil bool
		if v, isNil = indirect(v); isNil {
			return reflect.Value{}, fmt.Errorf("index of nil pointer")
		}
		switch v.Kind() {
		case reflect.Array, reflect.Slice, reflect.String:
			x, err := indexArg(index, v.Len())
			if err != nil {
				return reflect.Value{}, err
			}
			v = v.Index(x)
		case reflect.Map:
			index, err := prepareArg(index, v.Type().Key())
			if err != nil {
				return reflect.Value{}, err
			}
			if x := v.MapIndex(index); x.IsValid() {
				v = x
			} else {
				v = reflect.Zero(v.Type().Elem())
			}
		case reflect.Invalid:
			// the loop holds invariant: v.IsValid()
			panic("unreachable")
		default:
			return reflect.Value{}, fmt.Errorf("can't index item of type %s", v.Type())
		}
	}
	return v, nil
}

// Slicing.

// slice returns the result of slicing its first argument by the remaining
// arguments. Thus "slice x 1 2" is, in Go syntax, x[1:2], while "slice x"
// is x[:], "slice x 1" is x[1:], and "slice x 1 2 3" is x[1:2:3]. The first
// argument must be a string, slice, or array.
func slice(item reflect.Value, indexes ...reflect.Value) (reflect.Value, error) {
	var (
		cap int
		v   = indirectInterface(item)
	)
	if !v.IsValid() {
		return reflect.Value{}, fmt.Errorf("slice of untyped nil")
	}
	if len(indexes) > 3 {
		return reflect.Value{}, fmt.Errorf("too many slice indexes: %d", len(indexes))
	}
	switch v.Kind() {
	case reflect.String:
		if len(indexes) == 3 {
			return reflect.Value{}, fmt.Errorf("cannot 3-index slice a string")
		}
		cap = v.Len()
	case reflect.Array, reflect.Slice:
		cap = v.Cap()
	default:
		return reflect.Value{}, fmt.Errorf("can't slice item of type %s", v.Type())
	}
	// set default values for cases item[:], item[i:].
	idx := [3]int{0, v.Len()}
	for i, index := range indexes {
		x, err := indexArg(index, cap)
		if err != nil {
			return reflect.Value{}, err
		}
		idx[i] = x
	}
	// given item[i:j], make sure i <= j.
	if idx[0] > idx[1] {
		return reflect.Value{}, fmt.Errorf("invalid slice index: %d > %d", idx[0], idx[1])
	}
	if len(indexes) < 3 {
		return item.Slice(idx[0], idx[1]), nil
	}
	// given item[i:j:k], make sure i <= j <= k.
	if idx[1] > idx[2] {
		return reflect.Value{}, fmt.Errorf("invalid slice index: %d > %d", idx[1], idx[2])
	}
	return item.Slice3(idx[0], idx[1], idx[2]), nil
}

// Length

// length returns the length of the item, with an error if it has no defined length.
func length(item interface{}) (int, error) {
	v := reflect.ValueOf(item)
	if !v.IsValid() {
		return 0, fmt.Errorf("len of untyped nil")
	}
	v, isNil := indirect(v)
	if isNil {
		return 0, fmt.Errorf("len of nil pointer")
	}
	switch v.Kind() {
	case reflect.Array, reflect.Chan, reflect.Map, reflect.Slice, reflect.String:
		return v.Len(), nil
	}
	return 0, fmt.Errorf("len of type %s", v.Type())
}

// Function invocation

// call returns the result of evaluating the first argument as a function.
// The function must return 1 result, or 2 results, the second of which is an error.
func call(fn reflect.Value, args ...reflect.Value) (reflect.Value, error) {
	v := indirectInterface(fn)
	if !v.IsValid() {
		return reflect.Value{}, fmt.Errorf("call of nil")
	}
	typ := v.Type()
	if typ.Kind() != reflect.Func {
		return reflect.Value{}, fmt.Errorf("non-function of type %s", typ)
	}
	if !goodFunc(typ) {
		return reflect.Value{}, fmt.Errorf("function called with %d args; should be 1 or 2", typ.NumOut())
	}
	numIn := typ.NumIn()
	var dddType reflect.Type
	if typ.IsVariadic() {
		if len(args) < numIn-1 {
			return reflect.Value{}, fmt.Errorf("wrong number of args: got %d want at least %d", len(args), numIn-1)
		}
		dddType = typ.In(numIn - 1).Elem()
	} else {
		if len(args) != numIn {
			return reflect.Value{}, fmt.Errorf("wrong number of args: got %d want %d", len(args), numIn)
		}
	}
	argv := make([]reflect.Value, len(args))
	for i, arg := range args {
		value := indirectInterface(arg)
		// Compute the expected type. Clumsy because of variadics.
		argType := dddType
		if !typ.IsVariadic() || i < numIn-1 {
			argType = typ.In(i)
		}

		var err error
		if argv[i], err = prepareArg(value, argType); err != nil {
			return reflect.Value{}, fmt.Errorf("arg %d: %s", i, err)
		}
	}
	return safeCall(v, argv)
}

// safeCall runs fun.Call(args), and returns the resulting value and error, if
// any. If the call panics, the panic value is returned as an error.
func safeCall(fun reflect.Value, args []reflect.Value) (val reflect.Value, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(error); ok {
				err = e
			} else {
				err = fmt.Errorf("%v", r)
			}
		}
	}()
	ret := fun.Call(args)
	if len(ret) == 2 && !ret[1].IsNil() {
		return ret[0], ret[1].Interface().(error)
	}
	return ret[0], nil
}

// Boolean logic.

func truth(arg reflect.Value) bool {
	t, _ := isTrue(indirectInterface(arg))
	return t
}

// and computes the Boolean AND of its arguments, returning
// the first false argument it encounters, or the last argument.
func and(arg0 reflect.Value, args ...reflect.Value) reflect.Value {
	if !truth(arg0) {
		return arg0
	}
	for i := range args {
		arg0 = args[i]
		if !truth(arg0) {
			break
		}
	}
	return arg0
}

// or computes the Boolean OR of its arguments, returning
// the first true argument it encounters, or the last argument.
func or(arg0 reflect.Value, args ...reflect.Value) reflect.Value {
	if truth(arg0) {
		return arg0
	}
	for i := range args {
		arg0 = args[i]
		if truth(arg0) {
			break
		}
	}
	return arg0
}

// not returns the Boolean negation of its argument.
func not(arg reflect.Value) bool {
	return !truth(arg)
}

// Comparison.

// TODO: Perhaps allow comparison between signed and unsigned integers.

var (
	errBadComparisonType = errors.New("invalid type for comparison")
	errBadComparison     = errors.New("incompatible types for comparison")
	errNoComparison      = errors.New("missing argument for comparison")
)

type kind int

const (
	invalidKind kind = iota
	boolKind
	complexKind
	intKind
	floatKind
	stringKind
	uintKind
)

func basicKind(v reflect.Value) (kind, error) {
	switch v.Kind() {
	case reflect.Bool:
		return boolKind, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return intKind, nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return uintKind, nil
	case reflect.Float32, reflect.Float64:
		return floatKind, nil
	case reflect.Complex64, reflect.Complex128:
		return complexKind, nil
	case reflect.String:
		return stringKind, nil
	}
	return invalidKind, errBadComparisonType
}

// eq evaluates the comparison a == b || a == c || ...
func eq(arg1 reflect.Value, arg2 ...reflect.Value) (bool, error) {
	v1 := indirectInterface(arg1)
	k1, err := basicKind(v1)
	if err != nil {
		return false, err
	}
	if len(arg2) == 0 {
		return false, errNoComparison
	}
	for _, arg := range arg2 {
		v2 := indirectInterface(arg)
		k2, err := basicKind(v2)
		if err != nil {
			return false, err
		}
		truth := false
		if k1 != k2 {
			// Special case: Can compare integer values regardless of type's sign.
			switch {
			case k1 == intKind && k2 == uintKind:
				truth = v1.Int() >= 0 && uint64(v1.Int()) == v2.Uint()
			case k1 == uintKind && k2 == intKind:
				truth = v2.Int() >= 0 && v1.Uint() == uint64(v2.Int())
			default:
				return false, errBadComparison
			}
		} else {
			switch k1 {
			case boolKind:
				truth = v1.Bool() == v2.Bool()
			case complexKind:
				truth = v1.Complex() == v2.Complex()
			case floatKind:
				truth = v1.Float() == v2.Float()
			case intKind:
				truth = v1.Int() == v2.Int()
			case stringKind:
				truth = v1.String() == v2.String()
			case uintKind:
				truth = v1.Uint() == v2.Uint()
			default:
				panic("invalid kind")
			}
		}
		if truth {
			return true, nil
		}
	}
	return false, nil
}

// ne evaluates the comparison a != b.
func ne(arg1, arg2 reflect.Value) (bool, error) {
	// != is the inverse of ==.
	equal, err := eq(arg1, arg2)
	return !equal, err
}

// lt evaluates the comparison a < b.
func lt(arg1, arg2 reflect.Value) (bool, error) {
	v1 := indirectInterface(arg1)
	k1, err := basicKind(v1)
	if err != nil {
		return false, err
	}
	v2 := indirectInterface(arg2)
	k2, err := basicKind(v2)
	if err != nil {
		return false, err
	}
	truth := false
	if k1 != k2 {
		// Special case: Can compare integer values regardless of type's sign.
		switch {
		case k1 == intKind && k2 == uintKind:
			truth = v1.Int() < 0 || uint64(v1.Int()) < v2.Uint()
		case k1 == uintKind && k2 == intKind:
			truth = v2.Int() >= 0 && v1.Uint() < uint64(v2.Int())
		default:
			return false, errBadComparison
		}
	} else {
		switch k1 {
		case boolKind, complexKind:
			return false, errBadComparisonType
		case floatKind:
			truth = v1.Float() < v2.Float()
		case intKind:
			truth = v1.Int() < v2.Int()
		case stringKind:
			truth = v1.String() < v2.String()
		case uintKind:
			truth = v1.Uint() < v2.Uint()
		default:
			panic("invalid kind")
		}
	}
	return truth, nil
}

// le evaluates the comparison <= b.
func le(arg1, arg2 reflect.Value) (bool, error) {
	// <= is < or ==.
	lessThan, err := lt(arg1, arg2)
	if lessThan || err != nil {
		return lessThan, err
	}
	return eq(arg1, arg2)
}

// gt evaluates the comparison a > b.
func gt(arg1, arg2 reflect.Value) (bool, error) {
	// > is the inverse of <=.
	lessOrEqual, err := le(arg1, arg2)
	if err != nil {
		return false, err
	}
	return !lessOrEqual, nil
}

// ge evaluates the comparison a >= b.
func ge(arg1, arg2 reflect.Value) (bool, error) {
	// >= is the inverse of <.
	lessThan, err := lt(arg1, arg2)
	if err != nil {
		return false, err
	}
	return !lessThan, nil
}

// HTML escaping.

var (
	htmlQuot = []byte("&#34;") // shorter than "&quot;"
	htmlApos = []byte("&#39;") // shorter than "&apos;" and apos was not in HTML until HTML5
	htmlAmp  = []byte("&amp;")
	htmlLt   = []byte("&lt;")
	htmlGt   = []byte("&gt;")
	htmlNull = []byte("\uFFFD")
)

// HTMLEscape writes to w the escaped HTML equivalent of the plain text data b.
func HTMLEscape(w io.Writer, b []byte) {
	last := 0
	for i, c := range b {
		var html []byte
		switch c {
		case '\000':
			html = htmlNull
		case '"':
			html = htmlQuot
		case '\'':
			html = htmlApos
		case '&':
			html = htmlAmp
		case '<':
			html = htmlLt
		case '>':
			html = htmlGt
		default:
			continue
		}
		w.Write(b[last:i])
		w.Write(html)
		last = i + 1
	}
	w.Write(b[last:])
}

// HTMLEscapeString returns the escaped HTML equivalent of the plain text data s.
func HTMLEscapeString(s string) string {
	// Avoid allocation if we can.
	if !strings.ContainsAny(s, "'\"&<>\000") {
		return s
	}
	var b bytes.Buffer
	HTMLEscape(&b, []byte(s))
	return b.String()
}

// HTMLEscaper returns the escaped HTML equivalent of the textual
// representation of its arguments.
func HTMLEscaper(args ...interface{}) string {
	return HTMLEscapeString(evalArgs(args))
}

// JavaScript escaping.

var (
	jsLowUni = []byte(`\u00`)
	hex      = []byte("0123456789ABCDEF")

	jsBackslash = []byte(`\\`)
	jsApos      = []byte(`\'`)
	jsQuot      = []byte(`\"`)
	jsLt        = []byte(`\x3C`)
	jsGt        = []byte(`\x3E`)
)

// JSEscape writes to w the escaped JavaScript equivalent of the plain text data b.
func JSEscape(w io.Writer, b []byte) {
	last := 0
	for i := 0; i < len(b); i++ {
		c := b[i]

		if !jsIsSpecial(rune(c)) {
			// fast path: nothing to do
			continue
		}
		w.Write(b[last:i])

		if c < utf8.RuneSelf {
			// Quotes, slashes and angle brackets get quoted.
			// Control characters get written as \u00XX.
			switch c {
			case '\\':
				w.Write(jsBackslash)
			case '\'':
				w.Write(jsApos)
			case '"':
				w.Write(jsQuot)
			case '<':
				w.Write(jsLt)
			case '>':
				w.Write(jsGt)
			default:
				w.Write(jsLowUni)
				t, b := c>>4, c&0x0f
				w.Write(hex[t : t+1])
				w.Write(hex[b : b+1])
			}
		} else {
			// Unicode rune.
			r, size := utf8.DecodeRune(b[i:])
			if unicode.IsPrint(r) {
				w.Write(b[i : i+size])
			} else {
				fmt.Fprintf(w, "\\u%04X", r)
			}
			i += size - 1
		}
		last = i + 1
	}
	w.Write(b[last:])
}

// JSEscapeString returns the escaped JavaScript equivalent of the plain text data s.
func JSEscapeString(s string) string {
	// Avoid allocation if we can.
	if strings.IndexFunc(s, jsIsSpecial) < 0 {
		return s
	}
	var b bytes.Buffer
	JSEscape(&b, []byte(s))
	return b.String()
}

func jsIsSpecial(r rune) bool {
	switch r {
	case '\\', '\'', '"', '<', '>':
		return true
	}
	return r < ' ' || utf8.RuneSelf <= r
}

// JSEscaper returns the escaped JavaScript equivalent of the textual
// representation of its arguments.
func JSEscaper(args ...interface{}) string {
	return JSEscapeString(evalArgs(args))
}

// URLQueryEscaper returns the escaped value of the textual representation of
// its arguments in a form suitable for embedding in a URL query.
func URLQueryEscaper(args ...interface{}) string {
	return url.QueryEscape(evalArgs(args))
}

// evalArgs formats the list of arguments into a string. It is therefore equivalent to
//	fmt.Sprint(args...)
// except that each argument is indirected (if a pointer), as required,
// using the same rules as the default string evaluation during template
// execution.
func evalArgs(args []interface{}) string {
	ok := false
	var s string
	// Fast path for simple common case.
	if len(args) == 1 {
		s, ok = args[0].(string)
	}
	if !ok {
		for i, arg := range args {
			a, ok := printableValue(reflect.ValueOf(arg))
			if ok {
				args[i] = a
			} // else let fmt do its thing
		}
		s = fmt.Sprint(args...)
	}
	return s
}
