// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"io"
	"os"
	"reflect"
	"runtime"
	"strings"
	"template/parse"
)

// state represents the state of an execution. It's not part of the
// template so that multiple executions of the same template
// can execute in parallel.
type state struct {
	tmpl *Template
	wr   io.Writer
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

// setVar overwrites the top-nth variable on the stack. Used by range iterations.
func (s *state) setVar(n int, value reflect.Value) {
	s.vars[len(s.vars)-n].value = value
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
	format = fmt.Sprintf("template: %s:%d: %s", s.tmpl.Name(), s.line, format)
	panic(fmt.Errorf(format, args...))
}

// error terminates processing.
func (s *state) error(err os.Error) {
	s.errorf("%s", err)
}

// errRecover is the handler that turns panics into returns from the top
// level of Parse.
func errRecover(errp *os.Error) {
	e := recover()
	if e != nil {
		if _, ok := e.(runtime.Error); ok {
			panic(e)
		}
		*errp = e.(os.Error)
	}
}

// Execute applies a parsed template to the specified data object,
// writing the output to wr.
func (t *Template) Execute(wr io.Writer, data interface{}) (err os.Error) {
	defer errRecover(&err)
	value := reflect.ValueOf(data)
	state := &state{
		tmpl: t,
		wr:   wr,
		line: 1,
		vars: []variable{{"$", value}},
	}
	if t.Tree == nil || t.Root == nil {
		state.errorf("must be parsed before execution")
	}
	state.walk(value, t.Root)
	return
}

// Walk functions step through the major pieces of the template structure,
// generating output as they go.
func (s *state) walk(dot reflect.Value, n parse.Node) {
	switch n := n.(type) {
	case *parse.ActionNode:
		s.line = n.Line
		// Do not pop variables so they persist until next end.
		// Also, if the action declares variables, don't print the result.
		val := s.evalPipeline(dot, n.Pipe)
		if len(n.Pipe.Decl) == 0 {
			s.printValue(n, val)
		}
	case *parse.IfNode:
		s.line = n.Line
		s.walkIfOrWith(parse.NodeIf, dot, n.Pipe, n.List, n.ElseList)
	case *parse.ListNode:
		for _, node := range n.Nodes {
			s.walk(dot, node)
		}
	case *parse.RangeNode:
		s.line = n.Line
		s.walkRange(dot, n)
	case *parse.TemplateNode:
		s.line = n.Line
		s.walkTemplate(dot, n)
	case *parse.TextNode:
		if _, err := s.wr.Write(n.Text); err != nil {
			s.error(err)
		}
	case *parse.WithNode:
		s.line = n.Line
		s.walkIfOrWith(parse.NodeWith, dot, n.Pipe, n.List, n.ElseList)
	default:
		s.errorf("unknown node: %s", n)
	}
}

// walkIfOrWith walks an 'if' or 'with' node. The two control structures
// are identical in behavior except that 'with' sets dot.
func (s *state) walkIfOrWith(typ parse.NodeType, dot reflect.Value, pipe *parse.PipeNode, list, elseList *parse.ListNode) {
	defer s.pop(s.mark())
	val := s.evalPipeline(dot, pipe)
	truth, ok := isTrue(val)
	if !ok {
		s.errorf("if/with can't use %v", val)
	}
	if truth {
		if typ == parse.NodeWith {
			s.walk(val, list)
		} else {
			s.walk(dot, list)
		}
	} else if elseList != nil {
		s.walk(dot, elseList)
	}
}

// isTrue returns whether the value is 'true', in the sense of not the zero of its type,
// and whether the value has a meaningful truth value.
func isTrue(val reflect.Value) (truth, ok bool) {
	if !val.IsValid() {
		// Something like var x interface{}, never set. It's a form of nil.
		return false, true
	}
	switch val.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		truth = val.Len() > 0
	case reflect.Bool:
		truth = val.Bool()
	case reflect.Complex64, reflect.Complex128:
		truth = val.Complex() != 0
	case reflect.Chan, reflect.Func, reflect.Ptr, reflect.Interface:
		truth = !val.IsNil()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		truth = val.Int() != 0
	case reflect.Float32, reflect.Float64:
		truth = val.Float() != 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		truth = val.Uint() != 0
	case reflect.Struct:
		truth = true // Struct values are always true.
	default:
		return
	}
	return truth, true
}

func (s *state) walkRange(dot reflect.Value, r *parse.RangeNode) {
	defer s.pop(s.mark())
	val, _ := indirect(s.evalPipeline(dot, r.Pipe))
	// mark top of stack before any variables in the body are pushed.
	mark := s.mark()
	oneIteration := func(index, elem reflect.Value) {
		// Set top var (lexically the second if there are two) to the element.
		if len(r.Pipe.Decl) > 0 {
			s.setVar(1, elem)
		}
		// Set next var (lexically the first if there are two) to the index.
		if len(r.Pipe.Decl) > 1 {
			s.setVar(2, index)
		}
		s.walk(elem, r.List)
		s.pop(mark)
	}
	switch val.Kind() {
	case reflect.Array, reflect.Slice:
		if val.Len() == 0 {
			break
		}
		for i := 0; i < val.Len(); i++ {
			oneIteration(reflect.ValueOf(i), val.Index(i))
		}
		return
	case reflect.Map:
		if val.Len() == 0 {
			break
		}
		for _, key := range val.MapKeys() {
			oneIteration(key, val.MapIndex(key))
		}
		return
	case reflect.Chan:
		if val.IsNil() {
			break
		}
		i := 0
		for ; ; i++ {
			elem, ok := val.Recv()
			if !ok {
				break
			}
			oneIteration(reflect.ValueOf(i), elem)
		}
		if i == 0 {
			break
		}
		return
	case reflect.Invalid:
		break // An invalid value is likely a nil map, etc. and acts like an empty map.
	default:
		s.errorf("range can't iterate over %v", val)
	}
	if r.ElseList != nil {
		s.walk(dot, r.ElseList)
	}
}

func (s *state) walkTemplate(dot reflect.Value, t *parse.TemplateNode) {
	set := s.tmpl.set
	if set == nil {
		s.errorf("no set defined in which to invoke template named %q", t.Name)
	}
	tmpl := set.tmpl[t.Name]
	if tmpl == nil {
		s.errorf("template %q not in set", t.Name)
	}
	// Variables declared by the pipeline persist.
	dot = s.evalPipeline(dot, t.Pipe)
	newState := *s
	newState.tmpl = tmpl
	// No dynamic scoping: template invocations inherit no variables.
	newState.vars = []variable{{"$", dot}}
	newState.walk(dot, tmpl.Root)
}

// Eval functions evaluate pipelines, commands, and their elements and extract
// values from the data structure by examining fields, calling methods, and so on.
// The printing of those values happens only through walk functions.

// evalPipeline returns the value acquired by evaluating a pipeline. If the
// pipeline has a variable declaration, the variable will be pushed on the
// stack. Callers should therefore pop the stack after they are finished
// executing commands depending on the pipeline value.
func (s *state) evalPipeline(dot reflect.Value, pipe *parse.PipeNode) (value reflect.Value) {
	if pipe == nil {
		return
	}
	for _, cmd := range pipe.Cmds {
		value = s.evalCommand(dot, cmd, value) // previous value is this one's final arg.
		// If the object has type interface{}, dig down one level to the thing inside.
		if value.Kind() == reflect.Interface && value.Type().NumMethod() == 0 {
			value = reflect.ValueOf(value.Interface()) // lovely!
		}
	}
	for _, variable := range pipe.Decl {
		s.push(variable.Ident[0], value)
	}
	return value
}

func (s *state) notAFunction(args []parse.Node, final reflect.Value) {
	if len(args) > 1 || final.IsValid() {
		s.errorf("can't give argument to non-function %s", args[0])
	}
}

func (s *state) evalCommand(dot reflect.Value, cmd *parse.CommandNode, final reflect.Value) reflect.Value {
	firstWord := cmd.Args[0]
	switch n := firstWord.(type) {
	case *parse.FieldNode:
		return s.evalFieldNode(dot, n, cmd.Args, final)
	case *parse.IdentifierNode:
		// Must be a function.
		return s.evalFunction(dot, n.Ident, cmd.Args, final)
	case *parse.VariableNode:
		return s.evalVariableNode(dot, n, cmd.Args, final)
	}
	s.notAFunction(cmd.Args, final)
	switch word := firstWord.(type) {
	case *parse.BoolNode:
		return reflect.ValueOf(word.True)
	case *parse.DotNode:
		return dot
	case *parse.NumberNode:
		return s.idealConstant(word)
	case *parse.StringNode:
		return reflect.ValueOf(word.Text)
	}
	s.errorf("can't evaluate command %q", firstWord)
	panic("not reached")
}

// idealConstant is called to return the value of a number in a context where
// we don't know the type. In that case, the syntax of the number tells us
// its type, and we use Go rules to resolve.  Note there is no such thing as
// a uint ideal constant in this situation - the value must be of int type.
func (s *state) idealConstant(constant *parse.NumberNode) reflect.Value {
	// These are ideal constants but we don't know the type
	// and we have no context.  (If it was a method argument,
	// we'd know what we need.) The syntax guides us to some extent.
	switch {
	case constant.IsComplex:
		return reflect.ValueOf(constant.Complex128) // incontrovertible.
	case constant.IsFloat && strings.IndexAny(constant.Text, ".eE") >= 0:
		return reflect.ValueOf(constant.Float64)
	case constant.IsInt:
		n := int(constant.Int64)
		if int64(n) != constant.Int64 {
			s.errorf("%s overflows int", constant.Text)
		}
		return reflect.ValueOf(n)
	case constant.IsUint:
		s.errorf("%s overflows int", constant.Text)
	}
	return zero
}

func (s *state) evalFieldNode(dot reflect.Value, field *parse.FieldNode, args []parse.Node, final reflect.Value) reflect.Value {
	return s.evalFieldChain(dot, dot, field.Ident, args, final)
}

func (s *state) evalVariableNode(dot reflect.Value, v *parse.VariableNode, args []parse.Node, final reflect.Value) reflect.Value {
	// $x.Field has $x as the first ident, Field as the second. Eval the var, then the fields.
	value := s.varValue(v.Ident[0])
	if len(v.Ident) == 1 {
		return value
	}
	return s.evalFieldChain(dot, value, v.Ident[1:], args, final)
}

// evalFieldChain evaluates .X.Y.Z possibly followed by arguments.
// dot is the environment in which to evaluate arguments, while
// receiver is the value being walked along the chain.
func (s *state) evalFieldChain(dot, receiver reflect.Value, ident []string, args []parse.Node, final reflect.Value) reflect.Value {
	n := len(ident)
	for i := 0; i < n-1; i++ {
		receiver = s.evalField(dot, ident[i], nil, zero, receiver)
	}
	// Now if it's a method, it gets the arguments.
	return s.evalField(dot, ident[n-1], args, final, receiver)
}

func (s *state) evalFunction(dot reflect.Value, name string, args []parse.Node, final reflect.Value) reflect.Value {
	function, ok := findFunction(name, s.tmpl, s.tmpl.set)
	if !ok {
		s.errorf("%q is not a defined function", name)
	}
	return s.evalCall(dot, function, name, args, final)
}

// evalField evaluates an expression like (.Field) or (.Field arg1 arg2).
// The 'final' argument represents the return value from the preceding
// value of the pipeline, if any.
func (s *state) evalField(dot reflect.Value, fieldName string, args []parse.Node, final, receiver reflect.Value) reflect.Value {
	if !receiver.IsValid() {
		return zero
	}
	typ := receiver.Type()
	receiver, _ = indirect(receiver)
	// Unless it's an interface, need to get to a value of type *T to guarantee
	// we see all methods of T and *T.
	ptr := receiver
	if ptr.Kind() != reflect.Interface && ptr.CanAddr() {
		ptr = ptr.Addr()
	}
	if method, ok := methodByName(ptr, fieldName); ok {
		return s.evalCall(dot, method, fieldName, args, final)
	}
	hasArgs := len(args) > 1 || final.IsValid()
	// It's not a method; is it a field of a struct?
	receiver, isNil := indirect(receiver)
	if receiver.Kind() == reflect.Struct {
		tField, ok := receiver.Type().FieldByName(fieldName)
		if ok {
			field := receiver.FieldByIndex(tField.Index)
			if hasArgs {
				s.errorf("%s is not a method but has arguments", fieldName)
			}
			if tField.PkgPath == "" { // field is exported
				return field
			}
		}
	}
	// If it's a map, attempt to use the field name as a key.
	if receiver.Kind() == reflect.Map {
		nameVal := reflect.ValueOf(fieldName)
		if nameVal.Type().AssignableTo(receiver.Type().Key()) {
			if hasArgs {
				s.errorf("%s is not a method but has arguments", fieldName)
			}
			return receiver.MapIndex(nameVal)
		}
	}
	if isNil {
		s.errorf("nil pointer evaluating %s.%s", typ, fieldName)
	}
	s.errorf("can't evaluate field %s in type %s", fieldName, typ)
	panic("not reached")
}

// TODO: delete when reflect's own MethodByName is released.
func methodByName(receiver reflect.Value, name string) (reflect.Value, bool) {
	typ := receiver.Type()
	for i := 0; i < typ.NumMethod(); i++ {
		if typ.Method(i).Name == name {
			return receiver.Method(i), true // This value includes the receiver.
		}
	}
	return zero, false
}

var (
	osErrorType     = reflect.TypeOf((*os.Error)(nil)).Elem()
	fmtStringerType = reflect.TypeOf((*fmt.Stringer)(nil)).Elem()
)

// evalCall executes a function or method call. If it's a method, fun already has the receiver bound, so
// it looks just like a function call.  The arg list, if non-nil, includes (in the manner of the shell), arg[0]
// as the function itself.
func (s *state) evalCall(dot, fun reflect.Value, name string, args []parse.Node, final reflect.Value) reflect.Value {
	if args != nil {
		args = args[1:] // Zeroth arg is function name/node; not passed to function.
	}
	typ := fun.Type()
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
	// Args must be evaluated. Fixed args first.
	i := 0
	for ; i < numFixed; i++ {
		argv[i] = s.evalArg(dot, typ.In(i), args[i])
	}
	// Now the ... args.
	if typ.IsVariadic() {
		argType := typ.In(typ.NumIn() - 1).Elem() // Argument is a slice.
		for ; i < len(args); i++ {
			argv[i] = s.evalArg(dot, argType, args[i])
		}
	}
	// Add final value if necessary.
	if final.IsValid() {
		argv[i] = final
	}
	result := fun.Call(argv)
	// If we have an os.Error that is not nil, stop execution and return that error to the caller.
	if len(result) == 2 && !result[1].IsNil() {
		s.errorf("error calling %s: %s", name, result[1].Interface().(os.Error))
	}
	return result[0]
}

// validateType guarantees that the value is valid and assignable to the type.
func (s *state) validateType(value reflect.Value, typ reflect.Type) reflect.Value {
	if !value.IsValid() {
		s.errorf("invalid value; expected %s", typ)
	}
	if !value.Type().AssignableTo(typ) {
		// Does one dereference or indirection work? We could do more, as we
		// do with method receivers, but that gets messy and method receivers
		// are much more constrained, so it makes more sense there than here.
		// Besides, one is almost always all you need.
		switch {
		case value.Kind() == reflect.Ptr && value.Type().Elem().AssignableTo(typ):
			value = value.Elem()
		case reflect.PtrTo(value.Type()).AssignableTo(typ) && value.CanAddr():
			value = value.Addr()
		default:
			s.errorf("wrong type for value; expected %s; got %s", typ, value.Type())
		}
	}
	return value
}

func (s *state) evalArg(dot reflect.Value, typ reflect.Type, n parse.Node) reflect.Value {
	switch arg := n.(type) {
	case *parse.DotNode:
		return s.validateType(dot, typ)
	case *parse.FieldNode:
		return s.validateType(s.evalFieldNode(dot, arg, []parse.Node{n}, zero), typ)
	case *parse.VariableNode:
		return s.validateType(s.evalVariableNode(dot, arg, nil, zero), typ)
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
			return s.evalEmptyInterface(dot, n)
		}
	case reflect.String:
		return s.evalString(typ, n)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return s.evalUnsignedInteger(typ, n)
	}
	s.errorf("can't handle %s for arg of type %s", n, typ)
	panic("not reached")
}

func (s *state) evalBool(typ reflect.Type, n parse.Node) reflect.Value {
	if n, ok := n.(*parse.BoolNode); ok {
		value := reflect.New(typ).Elem()
		value.SetBool(n.True)
		return value
	}
	s.errorf("expected bool; found %s", n)
	panic("not reached")
}

func (s *state) evalString(typ reflect.Type, n parse.Node) reflect.Value {
	if n, ok := n.(*parse.StringNode); ok {
		value := reflect.New(typ).Elem()
		value.SetString(n.Text)
		return value
	}
	s.errorf("expected string; found %s", n)
	panic("not reached")
}

func (s *state) evalInteger(typ reflect.Type, n parse.Node) reflect.Value {
	if n, ok := n.(*parse.NumberNode); ok && n.IsInt {
		value := reflect.New(typ).Elem()
		value.SetInt(n.Int64)
		return value
	}
	s.errorf("expected integer; found %s", n)
	panic("not reached")
}

func (s *state) evalUnsignedInteger(typ reflect.Type, n parse.Node) reflect.Value {
	if n, ok := n.(*parse.NumberNode); ok && n.IsUint {
		value := reflect.New(typ).Elem()
		value.SetUint(n.Uint64)
		return value
	}
	s.errorf("expected unsigned integer; found %s", n)
	panic("not reached")
}

func (s *state) evalFloat(typ reflect.Type, n parse.Node) reflect.Value {
	if n, ok := n.(*parse.NumberNode); ok && n.IsFloat {
		value := reflect.New(typ).Elem()
		value.SetFloat(n.Float64)
		return value
	}
	s.errorf("expected float; found %s", n)
	panic("not reached")
}

func (s *state) evalComplex(typ reflect.Type, n parse.Node) reflect.Value {
	if n, ok := n.(*parse.NumberNode); ok && n.IsComplex {
		value := reflect.New(typ).Elem()
		value.SetComplex(n.Complex128)
		return value
	}
	s.errorf("expected complex; found %s", n)
	panic("not reached")
}

func (s *state) evalEmptyInterface(dot reflect.Value, n parse.Node) reflect.Value {
	switch n := n.(type) {
	case *parse.BoolNode:
		return reflect.ValueOf(n.True)
	case *parse.DotNode:
		return dot
	case *parse.FieldNode:
		return s.evalFieldNode(dot, n, nil, zero)
	case *parse.IdentifierNode:
		return s.evalFunction(dot, n.Ident, nil, zero)
	case *parse.NumberNode:
		return s.idealConstant(n)
	case *parse.StringNode:
		return reflect.ValueOf(n.Text)
	case *parse.VariableNode:
		return s.evalVariableNode(dot, n, nil, zero)
	}
	s.errorf("can't handle assignment of %s to empty interface argument", n)
	panic("not reached")
}

// indirect returns the item at the end of indirection, and a bool to indicate if it's nil.
// We indirect through pointers and empty interfaces (only) because
// non-empty interfaces have methods we might need.
func indirect(v reflect.Value) (rv reflect.Value, isNil bool) {
	for ; v.Kind() == reflect.Ptr || v.Kind() == reflect.Interface; v = v.Elem() {
		if v.IsNil() {
			return v, true
		}
		if v.Kind() == reflect.Interface && v.NumMethod() > 0 {
			break
		}
	}
	return v, false
}

// printValue writes the textual representation of the value to the output of
// the template.
func (s *state) printValue(n parse.Node, v reflect.Value) {
	if v.Kind() == reflect.Ptr {
		v, _ = indirect(v) // fmt.Fprint handles nil.
	}
	if !v.IsValid() {
		fmt.Fprint(s.wr, "<no value>")
		return
	}

	if !v.Type().Implements(fmtStringerType) {
		if v.CanAddr() && reflect.PtrTo(v.Type()).Implements(fmtStringerType) {
			v = v.Addr()
		} else {
			switch v.Kind() {
			case reflect.Chan, reflect.Func:
				s.errorf("can't print %s of type %s", n, v.Type())
			}
		}
	}
	fmt.Fprint(s.wr, v.Interface())
}
