// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"unicode"
	"utf8"
)

// state represents the state of an execution. It's not part of the
// template so that multiple executions of the same template
// can execute in parallel.
type state struct {
	tmpl *Template
	wr   io.Writer
	set  *Set
	line int        // line number for errors
	vars []variable // push-down stack of variable values.
}

// variable holds the dynamic value of a variable such as $, $x etc.
type variable struct {
	name  string
	value reflect.Value
}

// push pushes a new variable on the stack.
func (s *state) push(name string, value reflect.Value) {
	s.vars = append(s.vars, variable{name, value})
}

// mark returns the length of the variable stack.
func (s *state) mark() int {
	return len(s.vars)
}

// pop pops the variable stack up to the mark.
func (s *state) pop(mark int) {
	s.vars = s.vars[0:mark]
}

// setTop overwrites the top variable on the stack. Used by range iterations.
func (s *state) setTop(value reflect.Value) {
	s.vars[len(s.vars)-1].value = value
}

// varValue returns the value of the named variable.
func (s *state) varValue(name string) reflect.Value {
	for i := s.mark() - 1; i >= 0; i-- {
		if s.vars[i].name == name {
			return s.vars[i].value
		}
	}
	s.errorf("undefined variable: %s", name)
	return zero
}

var zero reflect.Value

// errorf formats the error and terminates processing.
func (s *state) errorf(format string, args ...interface{}) {
	format = fmt.Sprintf("template: %s:%d: %s", s.tmpl.name, s.line, format)
	panic(fmt.Errorf(format, args...))
}

// error terminates processing.
func (s *state) error(err os.Error) {
	s.errorf("%s", err)
}

// Execute applies a parsed template to the specified data object,
// writing the output to wr.
func (t *Template) Execute(wr io.Writer, data interface{}) os.Error {
	return t.ExecuteInSet(wr, data, nil)
}

// ExecuteInSet applies a parsed template to the specified data object,
// writing the output to wr. Nested template invocations will be resolved
// from the specified set.
func (t *Template) ExecuteInSet(wr io.Writer, data interface{}, set *Set) (err os.Error) {
	defer t.recover(&err)
	value := reflect.ValueOf(data)
	state := &state{
		tmpl: t,
		wr:   wr,
		set:  set,
		line: 1,
		vars: []variable{{"$", value}},
	}
	if t.root == nil {
		state.errorf("must be parsed before execution")
	}
	state.walk(value, t.root)
	if state.mark() != 1 {
		t.errorf("internal error: variable stack at %d", state.mark())
	}
	return
}

// Walk functions step through the major pieces of the template structure,
// generating output as they go.
func (s *state) walk(data reflect.Value, n node) {
	switch n := n.(type) {
	case *actionNode:
		s.line = n.line
		defer s.pop(s.mark())
		s.printValue(n, s.evalPipeline(data, n.pipe))
	case *ifNode:
		s.line = n.line
		s.walkIfOrWith(nodeIf, data, n.pipe, n.list, n.elseList)
	case *listNode:
		for _, node := range n.nodes {
			s.walk(data, node)
		}
	case *rangeNode:
		s.line = n.line
		s.walkRange(data, n)
	case *templateNode:
		s.line = n.line
		s.walkTemplate(data, n)
	case *textNode:
		if _, err := s.wr.Write(n.text); err != nil {
			s.error(err)
		}
	case *withNode:
		s.line = n.line
		s.walkIfOrWith(nodeWith, data, n.pipe, n.list, n.elseList)
	default:
		s.errorf("unknown node: %s", n)
	}
}

// walkIfOrWith walks an 'if' or 'with' node. The two control structures
// are identical in behavior except that 'with' sets dot.
func (s *state) walkIfOrWith(typ nodeType, data reflect.Value, pipe *pipeNode, list, elseList *listNode) {
	defer s.pop(s.mark())
	val := s.evalPipeline(data, pipe)
	truth, ok := isTrue(val)
	if !ok {
		s.errorf("if/with can't use value of type %T", val.Interface())
	}
	if truth {
		if typ == nodeWith {
			s.walk(val, list)
		} else {
			s.walk(data, list)
		}
	} else if elseList != nil {
		s.walk(data, elseList)
	}
}

// isTrue returns whether the value is 'true', in the sense of not the zero of its type,
// and whether the value has a meaningful truth value.
func isTrue(val reflect.Value) (truth, ok bool) {
	switch val.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		truth = val.Len() > 0
	case reflect.Bool:
		truth = val.Bool()
	case reflect.Complex64, reflect.Complex128:
		truth = val.Complex() != 0
	case reflect.Chan, reflect.Func, reflect.Ptr:
		truth = !val.IsNil()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		truth = val.Int() != 0
	case reflect.Float32, reflect.Float64:
		truth = val.Float() != 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		truth = val.Uint() != 0
	default:
		return
	}
	return truth, true
}

func (s *state) walkRange(data reflect.Value, r *rangeNode) {
	defer s.pop(s.mark())
	val, _ := indirect(s.evalPipeline(data, r.pipe))
	switch val.Kind() {
	case reflect.Array, reflect.Slice:
		if val.Len() == 0 {
			break
		}
		for i := 0; i < val.Len(); i++ {
			elem := val.Index(i)
			// Set $x to the element rather than the slice.
			if r.pipe.decl != nil {
				s.setTop(elem)
			}
			s.walk(elem, r.list)
		}
		return
	case reflect.Map:
		if val.Len() == 0 {
			break
		}
		for _, key := range val.MapKeys() {
			elem := val.MapIndex(key)
			// Set $x to the key rather than the map.
			if r.pipe.decl != nil {
				s.setTop(elem)
			}
			s.walk(elem, r.list)
		}
		return
	default:
		s.errorf("range can't iterate over value of type %T", val.Interface())
	}
	if r.elseList != nil {
		s.walk(data, r.elseList)
	}
}

func (s *state) walkTemplate(data reflect.Value, t *templateNode) {
	name := s.evalArg(data, reflect.TypeOf("string"), t.name).String()
	if s.set == nil {
		s.errorf("no set defined in which to invoke template named %q", name)
	}
	tmpl := s.set.tmpl[name]
	if tmpl == nil {
		s.errorf("template %q not in set", name)
	}
	defer s.pop(s.mark())
	data = s.evalPipeline(data, t.pipe)
	newState := *s
	newState.tmpl = tmpl
	// No dynamic scoping: template invocations inherit no variables.
	newState.vars = []variable{{"$", data}}
	newState.walk(data, tmpl.root)
}

// Eval functions evaluate pipelines, commands, and their elements and extract
// values from the data structure by examining fields, calling methods, and so on.
// The printing of those values happens only through walk functions.

// evalPipeline returns the value acquired by evaluating a pipeline. If the
// pipeline has a variable declaration, the variable will be pushed on the
// stack. Callers should therefore pop the stack after they are finished
// executing commands depending on the pipeline value.
func (s *state) evalPipeline(data reflect.Value, pipe *pipeNode) reflect.Value {
	value := zero
	for _, cmd := range pipe.cmds {
		value = s.evalCommand(data, cmd, value) // previous value is this one's final arg.
		// If the object has type interface{}, dig down one level to the thing inside.
		if value.Kind() == reflect.Interface && value.Type().NumMethod() == 0 {
			value = reflect.ValueOf(value.Interface()) // lovely!
		}
	}
	if pipe.decl != nil {
		s.push(pipe.decl.ident, value)
	}
	return value
}

func (s *state) notAFunction(args []node, final reflect.Value) {
	if len(args) > 1 || final.IsValid() {
		s.errorf("can't give argument to non-function %s", args[0])
	}
}

func (s *state) evalCommand(data reflect.Value, cmd *commandNode, final reflect.Value) reflect.Value {
	firstWord := cmd.args[0]
	switch n := firstWord.(type) {
	case *fieldNode:
		return s.evalFieldNode(data, n, cmd.args, final)
	case *identifierNode:
		// Must be a function.
		return s.evalFunction(data, n.ident, cmd.args, final)
	}
	s.notAFunction(cmd.args, final)
	switch word := firstWord.(type) {
	case *boolNode:
		return reflect.ValueOf(word.true)
	case *dotNode:
		return data
	case *numberNode:
		// These are ideal constants but we don't know the type
		// and we have no context.  (If it was a method argument,
		// we'd know what we need.) The syntax guides us to some extent.
		switch {
		case word.isComplex:
			return reflect.ValueOf(word.complex128) // incontrovertible.
		case word.isFloat && strings.IndexAny(word.text, ".eE") >= 0:
			return reflect.ValueOf(word.float64)
		case word.isInt:
			return reflect.ValueOf(word.int64)
		case word.isUint:
			return reflect.ValueOf(word.uint64)
		}
	case *stringNode:
		return reflect.ValueOf(word.text)
	case *variableNode:
		return s.varValue(word.ident)
	}
	s.errorf("can't handle command %q", firstWord)
	panic("not reached")
}

func (s *state) evalFieldNode(data reflect.Value, field *fieldNode, args []node, final reflect.Value) reflect.Value {
	// Up to the last entry, it must be a field.
	n := len(field.ident)
	for i := 0; i < n-1; i++ {
		data = s.evalField(data, field.ident[i], nil, zero, false)
	}
	// Now it can be a field or method and if a method, gets arguments.
	return s.evalField(data, field.ident[n-1], args, final, true)
}

func (s *state) evalFunction(data reflect.Value, name string, args []node, final reflect.Value) reflect.Value {
	function, ok := findFunction(name, s.tmpl, s.set)
	if !ok {
		s.errorf("%q is not a defined function", name)
	}
	return s.evalCall(data, function, name, false, args, final)
}

// Is this an exported - upper case - name?
func isExported(name string) bool {
	rune, _ := utf8.DecodeRuneInString(name)
	return unicode.IsUpper(rune)
}

// evalField evaluates an expression like (.Field) or (.Field arg1 arg2).
// The 'final' argument represents the return value from the preceding
// value of the pipeline, if any.
// If we're in a chain, such as (.X.Y.Z), .X and .Y cannot be methods;
// canBeMethod will be true only for the last element of such chains (here .Z).
func (s *state) evalField(data reflect.Value, fieldName string, args []node, final reflect.Value,
canBeMethod bool) reflect.Value {
	typ := data.Type()
	var isNil bool
	data, isNil = indirect(data)
	if canBeMethod {
		// Need to get to a value of type *T to guarantee we see all
		// methods of T and *T.
		ptr := data
		if ptr.CanAddr() {
			ptr = ptr.Addr()
		}
		if method, ok := methodByName(ptr.Type(), fieldName); ok {
			return s.evalCall(ptr, method.Func, fieldName, true, args, final)
		}
	}
	// It's not a method; is it a field of a struct?
	if data.Kind() == reflect.Struct {
		field := data.FieldByName(fieldName)
		if field.IsValid() {
			if len(args) > 1 || final.IsValid() {
				s.errorf("%s is not a method but has arguments", fieldName)
			}
			if isExported(fieldName) { // valid and exported
				return field
			}
		}
	}
	if isNil {
		s.errorf("nil pointer evaluating %s.%s", typ, fieldName)
	}
	s.errorf("can't handle evaluation of field %s in type %s", fieldName, typ)
	panic("not reached")
}

// TODO: delete when reflect's own MethodByName is released.
func methodByName(typ reflect.Type, name string) (reflect.Method, bool) {
	for i := 0; i < typ.NumMethod(); i++ {
		if typ.Method(i).Name == name {
			return typ.Method(i), true
		}
	}
	return reflect.Method{}, false
}

var (
	osErrorType = reflect.TypeOf(new(os.Error)).Elem()
)

func (s *state) evalCall(v, fun reflect.Value, name string, isMethod bool, args []node, final reflect.Value) reflect.Value {
	typ := fun.Type()
	if !isMethod && len(args) > 0 { // Args will be nil if it's a niladic call in an argument list
		args = args[1:] // first arg is name of function; not used in call.
	}
	numIn := len(args)
	if final.IsValid() {
		numIn++
	}
	numFixed := len(args)
	if typ.IsVariadic() {
		numFixed = typ.NumIn() - 1 // last arg is the variadic one.
		if numIn < numFixed {
			s.errorf("wrong number of args for %s: want at least %d got %d", name, typ.NumIn()-1, len(args))
		}
	} else if numIn < typ.NumIn()-1 || !typ.IsVariadic() && numIn != typ.NumIn() {
		s.errorf("wrong number of args for %s: want %d got %d", name, typ.NumIn(), len(args))
	}
	if !goodFunc(typ) {
		s.errorf("can't handle multiple results from method/function %q", name)
	}
	// Build the arg list.
	argv := make([]reflect.Value, numIn)
	// First arg is the receiver.
	i := 0
	if isMethod {
		argv[0] = v
		i++
	}
	// Others must be evaluated. Fixed args first.
	for ; i < numFixed; i++ {
		argv[i] = s.evalArg(v, typ.In(i), args[i])
	}
	// And now the ... args.
	if typ.IsVariadic() {
		argType := typ.In(typ.NumIn() - 1).Elem() // Argument is a slice.
		for ; i < len(args); i++ {
			argv[i] = s.evalArg(v, argType, args[i])
		}
	}
	// Add final value if necessary.
	if final.IsValid() {
		argv[len(args)] = final
	}
	result := fun.Call(argv)
	// If we have an os.Error that is not nil, stop execution and return that error to the caller.
	if len(result) == 2 && !result[1].IsNil() {
		s.error(result[1].Interface().(os.Error))
	}
	return result[0]
}

// validateType guarantees that the value is assignable to the type.
func (s *state) validateType(value reflect.Value, typ reflect.Type) reflect.Value {
	if !value.Type().AssignableTo(typ) {
		s.errorf("wrong type for value; expected %s; got %s", typ, value.Type())
	}
	return value
}

func (s *state) evalArg(data reflect.Value, typ reflect.Type, n node) reflect.Value {
	switch arg := n.(type) {
	case *dotNode:
		return s.validateType(data, typ)
	case *fieldNode:
		return s.validateType(s.evalFieldNode(data, arg, []node{n}, zero), typ)
	case *variableNode:
		return s.validateType(s.varValue(arg.ident), typ)
	}
	switch typ.Kind() {
	case reflect.Bool:
		return s.evalBool(typ, n)
	case reflect.Complex64, reflect.Complex128:
		return s.evalComplex(typ, n)
	case reflect.Float32, reflect.Float64:
		return s.evalFloat(typ, n)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return s.evalInteger(typ, n)
	case reflect.Interface:
		if typ.NumMethod() == 0 {
			return s.evalEmptyInterface(data, typ, n)
		}
	case reflect.String:
		return s.evalString(typ, n)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return s.evalUnsignedInteger(typ, n)
	}
	s.errorf("can't handle %s for arg of type %s", n, typ)
	panic("not reached")
}

func (s *state) evalBool(typ reflect.Type, n node) reflect.Value {
	if n, ok := n.(*boolNode); ok {
		value := reflect.New(typ).Elem()
		value.SetBool(n.true)
		return value
	}
	s.errorf("expected bool; found %s", n)
	panic("not reached")
}

func (s *state) evalString(typ reflect.Type, n node) reflect.Value {
	if n, ok := n.(*stringNode); ok {
		value := reflect.New(typ).Elem()
		value.SetString(n.text)
		return value
	}
	s.errorf("expected string; found %s", n)
	panic("not reached")
}

func (s *state) evalInteger(typ reflect.Type, n node) reflect.Value {
	if n, ok := n.(*numberNode); ok && n.isInt {
		value := reflect.New(typ).Elem()
		value.SetInt(n.int64)
		return value
	}
	s.errorf("expected integer; found %s", n)
	panic("not reached")
}

func (s *state) evalUnsignedInteger(typ reflect.Type, n node) reflect.Value {
	if n, ok := n.(*numberNode); ok && n.isUint {
		value := reflect.New(typ).Elem()
		value.SetUint(n.uint64)
		return value
	}
	s.errorf("expected unsigned integer; found %s", n)
	panic("not reached")
}

func (s *state) evalFloat(typ reflect.Type, n node) reflect.Value {
	if n, ok := n.(*numberNode); ok && n.isFloat {
		value := reflect.New(typ).Elem()
		value.SetFloat(n.float64)
		return value
	}
	s.errorf("expected float; found %s", n)
	panic("not reached")
}

func (s *state) evalComplex(typ reflect.Type, n node) reflect.Value {
	if n, ok := n.(*numberNode); ok && n.isComplex {
		value := reflect.New(typ).Elem()
		value.SetComplex(n.complex128)
		return value
	}
	s.errorf("expected complex; found %s", n)
	panic("not reached")
}

func (s *state) evalEmptyInterface(data reflect.Value, typ reflect.Type, n node) reflect.Value {
	switch n := n.(type) {
	case *boolNode:
		return reflect.ValueOf(n.true)
	case *dotNode:
		return data
	case *fieldNode:
		return s.evalFieldNode(data, n, nil, zero)
	case *identifierNode:
		return s.evalFunction(data, n.ident, nil, zero)
	case *numberNode:
		if n.isComplex {
			return reflect.ValueOf(n.complex128)
		}
		if n.isInt {
			return reflect.ValueOf(n.int64)
		}
		if n.isUint {
			return reflect.ValueOf(n.uint64)
		}
		if n.isFloat {
			return reflect.ValueOf(n.float64)
		}
	case *stringNode:
		return reflect.ValueOf(n.text)
	case *variableNode:
		return s.varValue(n.ident)
	}
	s.errorf("can't handle assignment of %s to empty interface argument", n)
	panic("not reached")
}

// indirect returns the item at the end of indirection, and a bool to indicate if it's nil.
func indirect(v reflect.Value) (rv reflect.Value, isNil bool) {
	for v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return v, true
		}
		v = v.Elem()
	}
	return v, false
}

// printValue writes the textual representation of the value to the output of
// the template.
func (s *state) printValue(n node, v reflect.Value) {
	if !v.IsValid() {
		fmt.Fprint(s.wr, "<no value>")
		return
	}
	switch v.Kind() {
	case reflect.Ptr:
		var isNil bool
		if v, isNil = indirect(v); isNil {
			fmt.Fprint(s.wr, "<nil>")
			return
		}
	case reflect.Chan, reflect.Func, reflect.Interface:
		s.errorf("can't print %s of type %s", n, v.Type())
	}
	fmt.Fprint(s.wr, v.Interface())
}
