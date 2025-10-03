// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"errors"
	"fmt"
	"io"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// FuncMap is the type of the map defining the mapping from names to functions.
// Each function must have either a single return value, or two return values of
// which the second has type error. In that case, if the second (error)
// return value evaluates to non-nil during execution, execution terminates and
// Execute returns that error.
//
// Errors returned by Execute wrap the underlying error; call [errors.As] to
// unwrap them.
//
// When template execution invokes a function with an argument list, that list
// must be assignable to the function's parameter types. Functions meant to
// apply to arguments of arbitrary type can use parameters of type interface{} or
// of type [reflect.Value]. Similarly, functions meant to return a result of arbitrary
// type can return interface{} or [reflect.Value].
type FuncMap map[string]any

// builtins returns the FuncMap.
// It is not a global variable so the linker can dead code eliminate
// more when this isn't called. See golang.org/issue/36021.
// TODO: revert this back to a global map once golang.org/issue/2559 is fixed.
func builtins() FuncMap {
	return FuncMap{
		"and":      and,
		"call":     emptyCall,
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
}

// builtinFuncs lazily computes & caches the builtinFuncs map.
var builtinFuncs = sync.OnceValue(func() map[string]reflect.Value {
	funcMap := builtins()
	m := make(map[string]reflect.Value, len(funcMap))
	addValueFuncs(m, funcMap)
	return m
})

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
		if err := goodFunc(name, v.Type()); err != nil {
			panic(err)
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
func goodFunc(name string, typ reflect.Type) error {
	// We allow functions with 1 result or 2 results where the second is an error.
	switch numOut := typ.NumOut(); {
	case numOut == 1:
		return nil
	case numOut == 2 && typ.Out(1) == errorType:
		return nil
	case numOut == 2:
		return fmt.Errorf("invalid function signature for %s: second return value should be error; is %s", name, typ.Out(1))
	default:
		return fmt.Errorf("function %s has %d return values; should be 1 or 2", name, typ.NumOut())
	}
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
func findFunction(name string, tmpl *Template) (v reflect.Value, isBuiltin, ok bool) {
	if tmpl != nil && tmpl.common != nil {
		tmpl.muFuncs.RLock()
		defer tmpl.muFuncs.RUnlock()
		if fn := tmpl.execFuncs[name]; fn.IsValid() {
			return fn, false, true
		}
	}
	if fn := builtinFuncs()[name]; fn.IsValid() {
		return fn, true, true
	}
	return reflect.Value{}, false, false
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
		return 0, errors.New("cannot index slice/array with nil")
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
	item = indirectInterface(item)
	if !item.IsValid() {
		return reflect.Value{}, errors.New("index of untyped nil")
	}
	for _, index := range indexes {
		index = indirectInterface(index)
		var isNil bool
		if item, isNil = indirect(item); isNil {
			return reflect.Value{}, errors.New("index of nil pointer")
		}
		switch item.Kind() {
		case reflect.Array, reflect.Slice, reflect.String:
			x, err := indexArg(index, item.Len())
			if err != nil {
				return reflect.Value{}, err
			}
			item = item.Index(x)
		case reflect.Map:
			index, err := prepareArg(index, item.Type().Key())
			if err != nil {
				return reflect.Value{}, err
			}
			if x := item.MapIndex(index); x.IsValid() {
				item = x
			} else {
				item = reflect.Zero(item.Type().Elem())
			}
		case reflect.Invalid:
			// the loop holds invariant: item.IsValid()
			panic("unreachable")
		default:
			return reflect.Value{}, fmt.Errorf("can't index item of type %s", item.Type())
		}
	}
	return item, nil
}

// Slicing.

// slice returns the result of slicing its first argument by the remaining
// arguments. Thus "slice x 1 2" is, in Go syntax, x[1:2], while "slice x"
// is x[:], "slice x 1" is x[1:], and "slice x 1 2 3" is x[1:2:3]. The first
// argument must be a string, slice, or array.
func slice(item reflect.Value, indexes ...reflect.Value) (reflect.Value, error) {
	item = indirectInterface(item)
	if !item.IsValid() {
		return reflect.Value{}, errors.New("slice of untyped nil")
	}
	if len(indexes) > 3 {
		return reflect.Value{}, fmt.Errorf("too many slice indexes: %d", len(indexes))
	}
	var cap int
	switch item.Kind() {
	case reflect.String:
		if len(indexes) == 3 {
			return reflect.Value{}, errors.New("cannot 3-index slice a string")
		}
		cap = item.Len()
	case reflect.Array, reflect.Slice:
		cap = item.Cap()
	default:
		return reflect.Value{}, fmt.Errorf("can't slice item of type %s", item.Type())
	}
	// set default values for cases item[:], item[i:].
	idx := [3]int{0, item.Len()}
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
func length(item reflect.Value) (int, error) {
	item, isNil := indirect(item)
	if isNil {
		return 0, errors.New("len of nil pointer")
	}
	switch item.Kind() {
	case reflect.Array, reflect.Chan, reflect.Map, reflect.Slice, reflect.String:
		return item.Len(), nil
	}
	return 0, fmt.Errorf("len of type %s", item.Type())
}

// Function invocation

func emptyCall(fn reflect.Value, args ...reflect.Value) reflect.Value {
	panic("unreachable") // implemented as a special case in evalCall
}

// call returns the result of evaluating the first argument as a function.
// The function must return 1 result, or 2 results, the second of which is an error.
func call(name string, fn reflect.Value, args ...reflect.Value) (reflect.Value, error) {
	fn = indirectInterface(fn)
	if !fn.IsValid() {
		return reflect.Value{}, errors.New("call of nil")
	}
	typ := fn.Type()
	if typ.Kind() != reflect.Func {
		return reflect.Value{}, fmt.Errorf("non-function %s of type %s", name, typ)
	}

	if err := goodFunc(name, typ); err != nil {
		return reflect.Value{}, err
	}
	numIn := typ.NumIn()
	var dddType reflect.Type
	if typ.IsVariadic() {
		if len(args) < numIn-1 {
			return reflect.Value{}, fmt.Errorf("wrong number of args for %s: got %d want at least %d", name, len(args), numIn-1)
		}
		dddType = typ.In(numIn - 1).Elem()
	} else {
		if len(args) != numIn {
			return reflect.Value{}, fmt.Errorf("wrong number of args for %s: got %d want %d", name, len(args), numIn)
		}
	}
	argv := make([]reflect.Value, len(args))
	for i, arg := range args {
		arg = indirectInterface(arg)
		// Compute the expected type. Clumsy because of variadics.
		argType := dddType
		if !typ.IsVariadic() || i < numIn-1 {
			argType = typ.In(i)
		}

		var err error
		if argv[i], err = prepareArg(arg, argType); err != nil {
			return reflect.Value{}, fmt.Errorf("arg %d: %w", i, err)
		}
	}
	return safeCall(fn, argv)
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
	panic("unreachable") // implemented as a special case in evalCall
}

// or computes the Boolean OR of its arguments, returning
// the first true argument it encounters, or the last argument.
func or(arg0 reflect.Value, args ...reflect.Value) reflect.Value {
	panic("unreachable") // implemented as a special case in evalCall
}

// not returns the Boolean negation of its argument.
func not(arg reflect.Value) bool {
	return !truth(arg)
}

// Comparison.

// TODO: Perhaps allow comparison between signed and unsigned integers.

var (
	errBadComparisonType = errors.New("invalid type for comparison")
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

// isNil returns true if v is the zero reflect.Value, or nil of its type.
func isNil(v reflect.Value) bool {
	if !v.IsValid() {
		return true
	}
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	}
	return false
}

// canCompare reports whether v1 and v2 are both the same kind, or one is nil.
// Called only when dealing with nillable types, or there's about to be an error.
func canCompare(v1, v2 reflect.Value) bool {
	k1 := v1.Kind()
	k2 := v2.Kind()
	if k1 == k2 {
		return true
	}
	// We know the type can be compared to nil.
	return k1 == reflect.Invalid || k2 == reflect.Invalid
}

// eq evaluates the comparison a == b || a == c || ...
func eq(arg1 reflect.Value, arg2 ...reflect.Value) (bool, error) {
	arg1 = indirectInterface(arg1)
	if len(arg2) == 0 {
		return false, errNoComparison
	}
	k1, _ := basicKind(arg1)
	for _, arg := range arg2 {
		arg = indirectInterface(arg)
		k2, _ := basicKind(arg)
		truth := false
		if k1 != k2 {
			// Special case: Can compare integer values regardless of type's sign.
			switch {
			case k1 == intKind && k2 == uintKind:
				truth = arg1.Int() >= 0 && uint64(arg1.Int()) == arg.Uint()
			case k1 == uintKind && k2 == intKind:
				truth = arg.Int() >= 0 && arg1.Uint() == uint64(arg.Int())
			default:
				if arg1.IsValid() && arg.IsValid() {
					return false, fmt.Errorf("incompatible types for comparison: %v and %v", arg1.Type(), arg.Type())
				}
			}
		} else {
			switch k1 {
			case boolKind:
				truth = arg1.Bool() == arg.Bool()
			case complexKind:
				truth = arg1.Complex() == arg.Complex()
			case floatKind:
				truth = arg1.Float() == arg.Float()
			case intKind:
				truth = arg1.Int() == arg.Int()
			case stringKind:
				truth = arg1.String() == arg.String()
			case uintKind:
				truth = arg1.Uint() == arg.Uint()
			default:
				if !canCompare(arg1, arg) {
					return false, fmt.Errorf("non-comparable types %s: %v, %s: %v", arg1, arg1.Type(), arg.Type(), arg)
				}
				if isNil(arg1) || isNil(arg) {
					truth = isNil(arg) == isNil(arg1)
				} else {
					if !arg.Type().Comparable() {
						return false, fmt.Errorf("non-comparable type %s: %v", arg, arg.Type())
					}
					truth = arg1.Interface() == arg.Interface()
				}
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
	arg1 = indirectInterface(arg1)
	k1, err := basicKind(arg1)
	if err != nil {
		return false, err
	}
	arg2 = indirectInterface(arg2)
	k2, err := basicKind(arg2)
	if err != nil {
		return false, err
	}
	truth := false
	if k1 != k2 {
		// Special case: Can compare integer values regardless of type's sign.
		switch {
		case k1 == intKind && k2 == uintKind:
			truth = arg1.Int() < 0 || uint64(arg1.Int()) < arg2.Uint()
		case k1 == uintKind && k2 == intKind:
			truth = arg2.Int() >= 0 && arg1.Uint() < uint64(arg2.Int())
		default:
			return false, fmt.Errorf("incompatible types for comparison: %v and %v", arg1.Type(), arg2.Type())
		}
	} else {
		switch k1 {
		case boolKind, complexKind:
			return false, errBadComparisonType
		case floatKind:
			truth = arg1.Float() < arg2.Float()
		case intKind:
			truth = arg1.Int() < arg2.Int()
		case stringKind:
			truth = arg1.String() < arg2.String()
		case uintKind:
			truth = arg1.Uint() < arg2.Uint()
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
	var b strings.Builder
	HTMLEscape(&b, []byte(s))
	return b.String()
}

// HTMLEscaper returns the escaped HTML equivalent of the textual
// representation of its arguments.
func HTMLEscaper(args ...any) string {
	return HTMLEscapeString(evalArgs(args))
}

// JavaScript escaping.

var (
	jsLowUni = []byte(`\u00`)
	hex      = []byte("0123456789ABCDEF")

	jsBackslash = []byte(`\\`)
	jsApos      = []byte(`\'`)
	jsQuot      = []byte(`\"`)
	jsLt        = []byte(`\u003C`)
	jsGt        = []byte(`\u003E`)
	jsAmp       = []byte(`\u0026`)
	jsEq        = []byte(`\u003D`)
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
			case '&':
				w.Write(jsAmp)
			case '=':
				w.Write(jsEq)
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
	var b strings.Builder
	JSEscape(&b, []byte(s))
	return b.String()
}

func jsIsSpecial(r rune) bool {
	switch r {
	case '\\', '\'', '"', '<', '>', '&', '=':
		return true
	}
	return r < ' ' || utf8.RuneSelf <= r
}

// JSEscaper returns the escaped JavaScript equivalent of the textual
// representation of its arguments.
func JSEscaper(args ...any) string {
	return JSEscapeString(evalArgs(args))
}

// URLQueryEscaper returns the escaped value of the textual representation of
// its arguments in a form suitable for embedding in a URL query.
func URLQueryEscaper(args ...any) string {
	return url.QueryEscape(evalArgs(args))
}

// evalArgs formats the list of arguments into a string. It is therefore equivalent to
//
//	fmt.Sprint(args...)
//
// except that each argument is indirected (if a pointer), as required,
// using the same rules as the default string evaluation during template
// execution.
func evalArgs(args []any) string {
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
