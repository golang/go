// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package demangle defines functions that demangle GCC/LLVM
// C++ and Rust symbol names.
// This package recognizes names that were mangled according to the C++ ABI
// defined at http://codesourcery.com/cxx-abi/ and the Rust ABI
// defined at
// https://rust-lang.github.io/rfcs/2603-rust-symbol-name-mangling-v0.html
//
// Most programs will want to call Filter or ToString.
package demangle

import (
	"errors"
	"fmt"
	"strings"
)

// ErrNotMangledName is returned by CheckedDemangle if the string does
// not appear to be a C++ symbol name.
var ErrNotMangledName = errors.New("not a C++ or Rust mangled name")

// Option is the type of demangler options.
type Option int

const (
	// The NoParams option disables demangling of function parameters.
	// It only omits the parameters of the function name being demangled,
	// not the parameter types of other functions that may be mentioned.
	// Using the option will speed up the demangler and cause it to
	// use less memory.
	NoParams Option = iota

	// The NoTemplateParams option disables demangling of template parameters.
	// This applies to both C++ and Rust.
	NoTemplateParams

	// The NoEnclosingParams option disables demangling of the function
	// parameter types of the enclosing function when demangling a
	// local name defined within a function.
	NoEnclosingParams

	// The NoClones option disables inclusion of clone suffixes.
	// NoParams implies NoClones.
	NoClones

	// The NoRust option disables demangling of old-style Rust
	// mangled names, which can be confused with C++ style mangled
	// names. New style Rust mangled names are still recognized.
	NoRust

	// The Verbose option turns on more verbose demangling.
	Verbose

	// LLVMStyle tries to translate an AST to a string in the
	// style of the LLVM demangler. This does not affect
	// the parsing of the AST, only the conversion of the AST
	// to a string.
	LLVMStyle
)

// maxLengthShift is how we shift the MaxLength value.
const maxLengthShift = 16

// maxLengthMask is a mask for the maxLength value.
const maxLengthMask = 0x1f << maxLengthShift

// MaxLength returns an Option that limits the maximum length of a
// demangled string. The maximum length is expressed as a power of 2,
// so a value of 1 limits the returned string to 2 characters, and
// a value of 16 limits the returned string to 65,536 characters.
// The value must be between 1 and 30.
func MaxLength(pow int) Option {
	if pow <= 0 || pow > 30 {
		panic("demangle: invalid MaxLength value")
	}
	return Option(pow << maxLengthShift)
}

// isMaxLength reports whether an Option holds a maximum length.
func isMaxLength(opt Option) bool {
	return opt&maxLengthMask != 0
}

// maxLength returns the maximum length stored in an Option.
func maxLength(opt Option) int {
	return 1 << ((opt & maxLengthMask) >> maxLengthShift)
}

// Filter demangles a C++ or Rust symbol name,
// returning the human-readable C++ or Rust name.
// If any error occurs during demangling, the input string is returned.
func Filter(name string, options ...Option) string {
	ret, err := ToString(name, options...)
	if err != nil {
		return name
	}
	return ret
}

// ToString demangles a C++ or Rust symbol name,
// returning a human-readable C++ or Rust name or an error.
// If the name does not appear to be a C++ or Rust symbol name at all,
// the error will be ErrNotMangledName.
func ToString(name string, options ...Option) (string, error) {
	if strings.HasPrefix(name, "_R") {
		return rustToString(name, options)
	}

	// Check for an old-style Rust mangled name.
	// It starts with _ZN and ends with "17h" followed by 16 hex digits
	// followed by "E" followed by an optional suffix starting with "."
	// (which we ignore).
	if strings.HasPrefix(name, "_ZN") {
		rname := name
		if pos := strings.LastIndex(rname, "E."); pos > 0 {
			rname = rname[:pos+1]
		}
		if strings.HasSuffix(rname, "E") && len(rname) > 23 && rname[len(rname)-20:len(rname)-17] == "17h" {
			noRust := false
			for _, o := range options {
				if o == NoRust {
					noRust = true
					break
				}
			}
			if !noRust {
				s, ok := oldRustToString(rname, options)
				if ok {
					return s, nil
				}
			}
		}
	}

	a, err := ToAST(name, options...)
	if err != nil {
		return "", err
	}
	return ASTToString(a, options...), nil
}

// ToAST demangles a C++ symbol name into an abstract syntax tree
// representing the symbol.
// If the NoParams option is passed, and the name has a function type,
// the parameter types are not demangled.
// If the name does not appear to be a C++ symbol name at all, the
// error will be ErrNotMangledName.
// This function does not currently support Rust symbol names.
func ToAST(name string, options ...Option) (AST, error) {
	if strings.HasPrefix(name, "_Z") {
		a, err := doDemangle(name[2:], options...)
		return a, adjustErr(err, 2)
	}

	if strings.HasPrefix(name, "___Z") {
		// clang extensions
		block := strings.LastIndex(name, "_block_invoke")
		if block == -1 {
			return nil, ErrNotMangledName
		}
		a, err := doDemangle(name[4:block], options...)
		if err != nil {
			return a, adjustErr(err, 4)
		}
		name = strings.TrimPrefix(name[block:], "_block_invoke")
		if len(name) > 0 && name[0] == '_' {
			name = name[1:]
		}
		for len(name) > 0 && isDigit(name[0]) {
			name = name[1:]
		}
		if len(name) > 0 && name[0] != '.' {
			return nil, errors.New("unparsed characters at end of mangled name")
		}
		a = &Special{Prefix: "invocation function for block in ", Val: a}
		return a, nil
	}

	const prefix = "_GLOBAL_"
	if strings.HasPrefix(name, prefix) {
		// The standard demangler ignores NoParams for global
		// constructors.  We are compatible.
		i := 0
		for i < len(options) {
			if options[i] == NoParams {
				options = append(options[:i], options[i+1:]...)
			} else {
				i++
			}
		}
		a, err := globalCDtorName(name[len(prefix):], options...)
		return a, adjustErr(err, len(prefix))
	}

	return nil, ErrNotMangledName
}

// globalCDtorName demangles a global constructor/destructor symbol name.
// The parameter is the string following the "_GLOBAL_" prefix.
func globalCDtorName(name string, options ...Option) (AST, error) {
	if len(name) < 4 {
		return nil, ErrNotMangledName
	}
	switch name[0] {
	case '.', '_', '$':
	default:
		return nil, ErrNotMangledName
	}

	var ctor bool
	switch name[1] {
	case 'I':
		ctor = true
	case 'D':
		ctor = false
	default:
		return nil, ErrNotMangledName
	}

	if name[2] != '_' {
		return nil, ErrNotMangledName
	}

	if !strings.HasPrefix(name[3:], "_Z") {
		return &GlobalCDtor{Ctor: ctor, Key: &Name{Name: name}}, nil
	} else {
		a, err := doDemangle(name[5:], options...)
		if err != nil {
			return nil, adjustErr(err, 5)
		}
		return &GlobalCDtor{Ctor: ctor, Key: a}, nil
	}
}

// The doDemangle function is the entry point into the demangler proper.
func doDemangle(name string, options ...Option) (ret AST, err error) {
	// When the demangling routines encounter an error, they panic
	// with a value of type demangleErr.
	defer func() {
		if r := recover(); r != nil {
			if de, ok := r.(demangleErr); ok {
				ret = nil
				err = de
				return
			}
			panic(r)
		}
	}()

	params := true
	clones := true
	verbose := false
	for _, o := range options {
		switch {
		case o == NoParams:
			params = false
			clones = false
		case o == NoClones:
			clones = false
		case o == Verbose:
			verbose = true
		case o == NoTemplateParams || o == NoEnclosingParams || o == LLVMStyle || isMaxLength(o):
			// These are valid options but only affect
			// printing of the AST.
		case o == NoRust:
			// Unimportant here.
		default:
			return nil, fmt.Errorf("unrecognized demangler option %v", o)
		}
	}

	st := &state{str: name, verbose: verbose}
	a := st.encoding(params, notForLocalName)

	// Accept a clone suffix.
	if clones {
		for len(st.str) > 1 && st.str[0] == '.' && (isLower(st.str[1]) || st.str[1] == '_' || isDigit(st.str[1])) {
			a = st.cloneSuffix(a)
		}
	}

	if clones && len(st.str) > 0 {
		st.fail("unparsed characters at end of mangled name")
	}

	return a, nil
}

// A state holds the current state of demangling a string.
type state struct {
	str       string        // remainder of string to demangle
	verbose   bool          // whether to use verbose demangling
	off       int           // offset of str within original string
	subs      substitutions // substitutions
	templates []*Template   // templates being processed

	// The number of entries in templates when we started parsing
	// a lambda, plus 1 so that 0 means not parsing a lambda.
	lambdaTemplateLevel int

	parsingConstraint bool // whether parsing a constraint expression

	// Counts of template parameters without template arguments,
	// for lambdas.
	typeTemplateParamCount     int
	nonTypeTemplateParamCount  int
	templateTemplateParamCount int
}

// copy returns a copy of the current state.
func (st *state) copy() *state {
	n := new(state)
	*n = *st
	return n
}

// fail panics with demangleErr, to be caught in doDemangle.
func (st *state) fail(err string) {
	panic(demangleErr{err: err, off: st.off})
}

// failEarlier is like fail, but decrements the offset to indicate
// that the point of failure occurred earlier in the string.
func (st *state) failEarlier(err string, dec int) {
	if st.off < dec {
		panic("internal error")
	}
	panic(demangleErr{err: err, off: st.off - dec})
}

// advance advances the current string offset.
func (st *state) advance(add int) {
	if len(st.str) < add {
		panic("internal error")
	}
	st.str = st.str[add:]
	st.off += add
}

// checkChar requires that the next character in the string be c, and
// advances past it.
func (st *state) checkChar(c byte) {
	if len(st.str) == 0 || st.str[0] != c {
		panic("internal error")
	}
	st.advance(1)
}

// A demangleErr is an error at a specific offset in the mangled
// string.
type demangleErr struct {
	err string
	off int
}

// Error implements the builtin error interface for demangleErr.
func (de demangleErr) Error() string {
	return fmt.Sprintf("%s at %d", de.err, de.off)
}

// adjustErr adjusts the position of err, if it is a demangleErr,
// and returns err.
func adjustErr(err error, adj int) error {
	if err == nil {
		return nil
	}
	if de, ok := err.(demangleErr); ok {
		de.off += adj
		return de
	}
	return err
}

type forLocalNameType int

const (
	forLocalName forLocalNameType = iota
	notForLocalName
)

// encoding parses:
//
//	encoding ::= <(function) name> <bare-function-type>
//	             <(data) name>
//	             <special-name>
func (st *state) encoding(params bool, local forLocalNameType) AST {
	if len(st.str) < 1 {
		st.fail("expected encoding")
	}

	if st.str[0] == 'G' || st.str[0] == 'T' {
		return st.specialName()
	}

	a, explicitObjectParameter := st.name()
	a = simplify(a)

	if !params {
		// Don't demangle the parameters.

		// Strip CV-qualifiers, as they apply to the 'this'
		// parameter, and are not output by the standard
		// demangler without parameters.
		if mwq, ok := a.(*MethodWithQualifiers); ok {
			a = mwq.Method
		}

		// If this is a local name, there may be CV-qualifiers
		// on the name that really apply to the top level, and
		// therefore must be discarded when discarding
		// parameters.  This can happen when parsing a class
		// that is local to a function.
		if q, ok := a.(*Qualified); ok && q.LocalName {
			p := &q.Name
			if da, ok := (*p).(*DefaultArg); ok {
				p = &da.Arg
			}
			if mwq, ok := (*p).(*MethodWithQualifiers); ok {
				*p = mwq.Method
			}
		}

		return a
	}

	if len(st.str) == 0 || st.str[0] == 'E' {
		// There are no parameters--this is a data symbol, not
		// a function symbol.
		return a
	}

	mwq, _ := a.(*MethodWithQualifiers)

	var findTemplate func(AST) *Template
	findTemplate = func(check AST) *Template {
		switch check := check.(type) {
		case *Template:
			return check
		case *Qualified:
			if check.LocalName {
				return findTemplate(check.Name)
			} else if _, ok := check.Name.(*Constructor); ok {
				return findTemplate(check.Name)
			}
		case *MethodWithQualifiers:
			return findTemplate(check.Method)
		case *Constructor:
			if check.Base != nil {
				return findTemplate(check.Base)
			}
		}
		return nil
	}

	template := findTemplate(a)
	var oldLambdaTemplateLevel int
	if template != nil {
		st.templates = append(st.templates, template)
		oldLambdaTemplateLevel = st.lambdaTemplateLevel
		st.lambdaTemplateLevel = 0
	}

	// Checking for the enable_if attribute here is what the LLVM
	// demangler does.  This is not very general but perhaps it is
	// sufficient.
	const enableIfPrefix = "Ua9enable_ifI"
	var enableIfArgs []AST
	if strings.HasPrefix(st.str, enableIfPrefix) {
		st.advance(len(enableIfPrefix) - 1)
		enableIfArgs = st.templateArgs()
	}

	ft := st.bareFunctionType(hasReturnType(a), explicitObjectParameter)

	var constraint AST
	if len(st.str) > 0 && st.str[0] == 'Q' {
		constraint = st.constraintExpr()
	}

	if template != nil {
		st.templates = st.templates[:len(st.templates)-1]
		st.lambdaTemplateLevel = oldLambdaTemplateLevel
	}

	ft = simplify(ft)

	// For a local name, discard the return type, so that it
	// doesn't get confused with the top level return type.
	if local == forLocalName {
		if functype, ok := ft.(*FunctionType); ok {
			functype.ForLocalName = true
		}
	}

	// Any top-level qualifiers belong to the function type.
	if mwq != nil {
		a = mwq.Method
		mwq.Method = ft
		ft = mwq
	}
	if q, ok := a.(*Qualified); ok && q.LocalName {
		p := &q.Name
		if da, ok := (*p).(*DefaultArg); ok {
			p = &da.Arg
		}
		if mwq, ok := (*p).(*MethodWithQualifiers); ok {
			*p = mwq.Method
			mwq.Method = ft
			ft = mwq
		}
	}

	r := AST(&Typed{Name: a, Type: ft})

	if len(enableIfArgs) > 0 {
		r = &EnableIf{Type: r, Args: enableIfArgs}
	}

	if constraint != nil {
		r = &Constraint{Name: r, Requires: constraint}
	}

	return r
}

// hasReturnType returns whether the mangled form of a will have a
// return type.
func hasReturnType(a AST) bool {
	switch a := a.(type) {
	case *Qualified:
		if a.LocalName {
			return hasReturnType(a.Name)
		}
		return false
	case *Template:
		return !isCDtorConversion(a.Name)
	case *TypeWithQualifiers:
		return hasReturnType(a.Base)
	case *MethodWithQualifiers:
		return hasReturnType(a.Method)
	default:
		return false
	}
}

// isCDtorConversion returns when an AST is a constructor, a
// destructor, or a conversion operator.
func isCDtorConversion(a AST) bool {
	switch a := a.(type) {
	case *Qualified:
		return isCDtorConversion(a.Name)
	case *Constructor, *Destructor, *Cast:
		return true
	default:
		return false
	}
}

// taggedName parses:
//
//	<tagged-name> ::= <name> B <source-name>
func (st *state) taggedName(a AST) AST {
	for len(st.str) > 0 && st.str[0] == 'B' {
		st.advance(1)
		tag := st.sourceName()
		a = &TaggedName{Name: a, Tag: tag}
	}
	return a
}

// name parses:
//
//	<name> ::= <nested-name>
//	       ::= <unscoped-name>
//	       ::= <unscoped-template-name> <template-args>
//	       ::= <local-name>
//
//	<unscoped-name> ::= <unqualified-name>
//	                ::= St <unqualified-name>
//
//	<unscoped-template-name> ::= <unscoped-name>
//	                         ::= <substitution>
//
// Besides the name, this returns whether it saw the code indicating
// a C++23 explicit object parameter.
func (st *state) name() (AST, bool) {
	if len(st.str) < 1 {
		st.fail("expected name")
	}

	var module AST
	switch st.str[0] {
	case 'N':
		return st.nestedName()
	case 'Z':
		return st.localName()
	case 'U':
		a, isCast := st.unqualifiedName(nil)
		if isCast {
			st.setTemplate(a, nil)
		}
		return a, false
	case 'S':
		if len(st.str) < 2 {
			st.advance(1)
			st.fail("expected substitution index")
		}
		var a AST
		isCast := false
		subst := false
		if st.str[1] == 't' {
			st.advance(2)
			a, isCast = st.unqualifiedName(nil)
			a = &Qualified{Scope: &Name{Name: "std"}, Name: a, LocalName: false}
		} else {
			a = st.substitution(false)
			if mn, ok := a.(*ModuleName); ok {
				module = mn
				break
			}
			subst = true
		}
		if len(st.str) > 0 && st.str[0] == 'I' {
			// This can only happen if we saw
			// <unscoped-template-name> and are about to see
			// <template-args>.  <unscoped-template-name> is a
			// substitution candidate if it did not come from a
			// substitution.
			if !subst {
				st.subs.add(a)
			}
			args := st.templateArgs()
			tmpl := &Template{Name: a, Args: args}
			if isCast {
				st.setTemplate(a, tmpl)
				st.clearTemplateArgs(args)
				isCast = false
			}
			a = tmpl
		}
		if isCast {
			st.setTemplate(a, nil)
		}
		return a, false
	}

	a, isCast := st.unqualifiedName(module)
	if len(st.str) > 0 && st.str[0] == 'I' {
		st.subs.add(a)
		args := st.templateArgs()
		tmpl := &Template{Name: a, Args: args}
		if isCast {
			st.setTemplate(a, tmpl)
			st.clearTemplateArgs(args)
			isCast = false
		}
		a = tmpl
	}
	if isCast {
		st.setTemplate(a, nil)
	}
	return a, false
}

// nestedName parses:
//
//	<nested-name> ::= N [<CV-qualifiers>] [<ref-qualifier>] <prefix> <unqualified-name> E
//	              ::= N [<CV-qualifiers>] [<ref-qualifier>] <template-prefix> <template-args> E
//
// Besides the name, this returns whether it saw the code indicating
// a C++23 explicit object parameter.
func (st *state) nestedName() (AST, bool) {
	st.checkChar('N')

	var q AST
	var r string

	explicitObjectParameter := false
	if len(st.str) > 0 && st.str[0] == 'H' {
		st.advance(1)
		explicitObjectParameter = true
	} else {
		q = st.cvQualifiers()
		r = st.refQualifier()
	}

	a := st.prefix()

	if q != nil || r != "" {
		a = &MethodWithQualifiers{Method: a, Qualifiers: q, RefQualifier: r}
	}
	if len(st.str) == 0 || st.str[0] != 'E' {
		st.fail("expected E after nested name")
	}
	st.advance(1)
	return a, explicitObjectParameter
}

// prefix parses:
//
//	<prefix> ::= <prefix> <unqualified-name>
//	         ::= <template-prefix> <template-args>
//	         ::= <template-param>
//	         ::= <decltype>
//	         ::=
//	         ::= <substitution>
//
//	<template-prefix> ::= <prefix> <(template) unqualified-name>
//	                  ::= <template-param>
//	                  ::= <substitution>
//
//	<decltype> ::= Dt <expression> E
//	           ::= DT <expression> E
func (st *state) prefix() AST {
	var a AST

	// The last name seen, for a constructor/destructor.
	var last AST

	var module AST

	getLast := func(a AST) AST {
		for {
			if t, ok := a.(*Template); ok {
				a = t.Name
			} else if q, ok := a.(*Qualified); ok {
				a = q.Name
			} else if t, ok := a.(*TaggedName); ok {
				a = t.Name
			} else {
				return a
			}
		}
	}

	var cast *Cast
	for {
		if len(st.str) == 0 {
			st.fail("expected prefix")
		}
		var next AST

		c := st.str[0]
		if isDigit(c) || isLower(c) || c == 'U' || c == 'L' || c == 'F' || c == 'W' || (c == 'D' && len(st.str) > 1 && st.str[1] == 'C') {
			un, isUnCast := st.unqualifiedName(module)
			next = un
			module = nil
			if isUnCast {
				if tn, ok := un.(*TaggedName); ok {
					un = tn.Name
				}
				cast = un.(*Cast)
			}
		} else {
			switch st.str[0] {
			case 'C':
				inheriting := false
				st.advance(1)
				if len(st.str) > 0 && st.str[0] == 'I' {
					inheriting = true
					st.advance(1)
				}
				if len(st.str) < 1 {
					st.fail("expected constructor type")
				}
				if last == nil {
					st.fail("constructor before name is seen")
				}
				st.advance(1)
				var base AST
				if inheriting {
					base = st.demangleType(false)
				}
				next = &Constructor{
					Name: getLast(last),
					Base: base,
				}
				if len(st.str) > 0 && st.str[0] == 'B' {
					next = st.taggedName(next)
				}
			case 'D':
				if len(st.str) > 1 && (st.str[1] == 'T' || st.str[1] == 't') {
					next = st.demangleType(false)
				} else {
					if len(st.str) < 2 {
						st.fail("expected destructor type")
					}
					if last == nil {
						st.fail("destructor before name is seen")
					}
					st.advance(2)
					next = &Destructor{Name: getLast(last)}
					if len(st.str) > 0 && st.str[0] == 'B' {
						next = st.taggedName(next)
					}
				}
			case 'S':
				next = st.substitution(true)
				if mn, ok := next.(*ModuleName); ok {
					module = mn
					next = nil
				}
			case 'I':
				if a == nil {
					st.fail("unexpected template arguments")
				}
				var args []AST
				args = st.templateArgs()
				tmpl := &Template{Name: a, Args: args}
				if cast != nil {
					st.setTemplate(cast, tmpl)
					st.clearTemplateArgs(args)
					cast = nil
				}
				a = nil
				next = tmpl
			case 'T':
				next = st.templateParam()
			case 'E':
				if a == nil {
					st.fail("expected prefix")
				}
				if cast != nil {
					var toTmpl *Template
					if castTempl, ok := cast.To.(*Template); ok {
						toTmpl = castTempl
					}
					st.setTemplate(cast, toTmpl)
				}
				return a
			case 'M':
				if a == nil {
					st.fail("unexpected lambda initializer")
				}
				// This is the initializer scope for a
				// lambda.  We don't need to record
				// it.  The normal code will treat the
				// variable has a type scope, which
				// gives appropriate output.
				st.advance(1)
				continue
			case 'J':
				// It appears that in some cases clang
				// can emit a J for a template arg
				// without the expected I.  I don't
				// know when this happens, but I've
				// seen it in some large C++ programs.
				if a == nil {
					st.fail("unexpected template arguments")
				}
				var args []AST
				for len(st.str) == 0 || st.str[0] != 'E' {
					arg := st.templateArg(nil)
					args = append(args, arg)
				}
				st.advance(1)
				tmpl := &Template{Name: a, Args: args}
				if cast != nil {
					st.setTemplate(cast, tmpl)
					st.clearTemplateArgs(args)
					cast = nil
				}
				a = nil
				next = tmpl
			default:
				st.fail("unrecognized letter in prefix")
			}
		}

		if next == nil {
			continue
		}

		last = next
		if a == nil {
			a = next
		} else {
			a = &Qualified{Scope: a, Name: next, LocalName: false}
		}

		if c != 'S' && (len(st.str) == 0 || st.str[0] != 'E') {
			st.subs.add(a)
		}
	}
}

// unqualifiedName parses:
//
//	<unqualified-name> ::= <operator-name>
//	                   ::= <ctor-dtor-name>
//	                   ::= <source-name>
//	                   ::= <local-source-name>
//
//	 <local-source-name>	::= L <source-name> <discriminator>
func (st *state) unqualifiedName(module AST) (r AST, isCast bool) {
	if len(st.str) < 1 {
		st.fail("expected unqualified name")
	}

	module = st.moduleName(module)

	friend := false
	if len(st.str) > 0 && st.str[0] == 'F' {
		st.advance(1)
		friend = true
	}

	var a AST
	isCast = false
	c := st.str[0]
	if isDigit(c) {
		a = st.sourceName()
	} else if isLower(c) {
		a, _ = st.operatorName(false)
		if _, ok := a.(*Cast); ok {
			isCast = true
		}
		if op, ok := a.(*Operator); ok && op.Name == `operator"" ` {
			n := st.sourceName()
			a = &Unary{Op: op, Expr: n, Suffix: false, SizeofType: false}
		}
	} else if c == 'D' && len(st.str) > 1 && st.str[1] == 'C' {
		var bindings []AST
		st.advance(2)
		for {
			binding := st.sourceName()
			bindings = append(bindings, binding)
			if len(st.str) > 0 && st.str[0] == 'E' {
				st.advance(1)
				break
			}
		}
		a = &StructuredBindings{Bindings: bindings}
	} else {
		switch c {
		case 'C', 'D':
			st.fail("constructor/destructor not in nested name")
		case 'L':
			st.advance(1)
			a = st.sourceName()
			a = st.discriminator(a)
		case 'U':
			if len(st.str) < 2 {
				st.advance(1)
				st.fail("expected closure or unnamed type")
			}
			c := st.str[1]
			switch c {
			case 'b':
				st.advance(2)
				st.compactNumber()
				a = &Name{Name: "'block-literal'"}
			case 'l':
				a = st.closureTypeName()
			case 't':
				a = st.unnamedTypeName()
			default:
				st.advance(1)
				st.fail("expected closure or unnamed type")
			}
		default:
			st.fail("expected unqualified name")
		}
	}

	if module != nil {
		a = &ModuleEntity{Module: module, Name: a}
	}

	if len(st.str) > 0 && st.str[0] == 'B' {
		a = st.taggedName(a)
	}

	if friend {
		a = &Friend{Name: a}
	}

	return a, isCast
}

// sourceName parses:
//
//	<source-name> ::= <(positive length) number> <identifier>
//	identifier ::= <(unqualified source code identifier)>
func (st *state) sourceName() AST {
	val := st.number()
	if val <= 0 {
		st.fail("expected positive number")
	}
	if len(st.str) < val {
		st.fail("not enough characters for identifier")
	}
	id := st.str[:val]
	st.advance(val)

	// Look for GCC encoding of anonymous namespace, and make it
	// more friendly.
	const anonPrefix = "_GLOBAL_"
	if strings.HasPrefix(id, anonPrefix) && len(id) > len(anonPrefix)+2 {
		c1 := id[len(anonPrefix)]
		c2 := id[len(anonPrefix)+1]
		if (c1 == '.' || c1 == '_' || c1 == '$') && c2 == 'N' {
			id = "(anonymous namespace)"
		}
	}

	n := &Name{Name: id}
	return n
}

// moduleName parses:
//
//	<module-name> ::= <module-subname>
//	 	      ::= <module-name> <module-subname>
//		      ::= <substitution>  # passed in by caller
//	<module-subname> ::= W <source-name>
//			 ::= W P <source-name>
//
// The module name is optional. If it is not present, this returns the parent.
func (st *state) moduleName(parent AST) AST {
	ret := parent
	for len(st.str) > 0 && st.str[0] == 'W' {
		st.advance(1)
		isPartition := false
		if len(st.str) > 0 && st.str[0] == 'P' {
			st.advance(1)
			isPartition = true
		}
		name := st.sourceName()
		ret = &ModuleName{
			Parent:      ret,
			Name:        name,
			IsPartition: isPartition,
		}
		st.subs.add(ret)
	}
	return ret
}

// number parses:
//
//	number ::= [n] <(non-negative decimal integer)>
func (st *state) number() int {
	neg := false
	if len(st.str) > 0 && st.str[0] == 'n' {
		neg = true
		st.advance(1)
	}
	if len(st.str) == 0 || !isDigit(st.str[0]) {
		st.fail("missing number")
	}
	val := 0
	for len(st.str) > 0 && isDigit(st.str[0]) {
		// Number picked to ensure we can't overflow with 32-bit int.
		// Any very large number here is bogus.
		if val >= 0x80000000/10-10 {
			st.fail("numeric overflow")
		}
		val = val*10 + int(st.str[0]-'0')
		st.advance(1)
	}
	if neg {
		val = -val
	}
	return val
}

// seqID parses:
//
//	<seq-id> ::= <0-9A-Z>+
//
// We expect this to be followed by an underscore.
func (st *state) seqID(eofOK bool) int {
	if len(st.str) > 0 && st.str[0] == '_' {
		st.advance(1)
		return 0
	}
	id := 0
	for {
		if len(st.str) == 0 {
			if eofOK {
				return id + 1
			}
			st.fail("missing end to sequence ID")
		}
		// Don't overflow a 32-bit int.
		if id >= 0x80000000/36-36 {
			st.fail("sequence ID overflow")
		}
		c := st.str[0]
		if c == '_' {
			st.advance(1)
			return id + 1
		}
		if isDigit(c) {
			id = id*36 + int(c-'0')
		} else if isUpper(c) {
			id = id*36 + int(c-'A') + 10
		} else {
			st.fail("invalid character in sequence ID")
		}
		st.advance(1)
	}
}

// An operator is the demangled name, and the number of arguments it
// takes in an expression.
type operator struct {
	name string
	args int
	prec precedence
}

// The operators map maps the mangled operator names to information
// about them.
var operators = map[string]operator{
	"aN": {"&=", 2, precAssign},
	"aS": {"=", 2, precAssign},
	"aa": {"&&", 2, precLogicalAnd},
	"ad": {"&", 1, precUnary},
	"an": {"&", 2, precAnd},
	"at": {"alignof ", 1, precUnary},
	"aw": {"co_await ", 1, precPrimary},
	"az": {"alignof ", 1, precUnary},
	"cc": {"const_cast", 2, precPostfix},
	"cl": {"()", 2, precPostfix},
	// cp is not in the ABI but is used by clang "when the call
	// would use ADL except for being parenthesized."
	"cp": {"()", 2, precPostfix},
	"cm": {",", 2, precComma},
	"co": {"~", 1, precUnary},
	"dV": {"/=", 2, precAssign},
	"dX": {"[...]=", 3, precAssign},
	"da": {"delete[] ", 1, precUnary},
	"dc": {"dynamic_cast", 2, precPostfix},
	"de": {"*", 1, precUnary},
	"di": {"=", 2, precAssign},
	"dl": {"delete ", 1, precUnary},
	"ds": {".*", 2, precPtrMem},
	"dt": {".", 2, precPostfix},
	"dv": {"/", 2, precAssign},
	"dx": {"]=", 2, precAssign},
	"eO": {"^=", 2, precAssign},
	"eo": {"^", 2, precXor},
	"eq": {"==", 2, precEqual},
	"fl": {"...", 2, precPrimary},
	"fr": {"...", 2, precPrimary},
	"fL": {"...", 3, precPrimary},
	"fR": {"...", 3, precPrimary},
	"ge": {">=", 2, precRel},
	"gs": {"::", 1, precUnary},
	"gt": {">", 2, precRel},
	"ix": {"[]", 2, precPostfix},
	"lS": {"<<=", 2, precAssign},
	"le": {"<=", 2, precRel},
	"li": {`operator"" `, 1, precUnary},
	"ls": {"<<", 2, precShift},
	"lt": {"<", 2, precRel},
	"mI": {"-=", 2, precAssign},
	"mL": {"*=", 2, precAssign},
	"mi": {"-", 2, precAdd},
	"ml": {"*", 2, precMul},
	"mm": {"--", 1, precPostfix},
	"na": {"new[]", 3, precUnary},
	"ne": {"!=", 2, precEqual},
	"ng": {"-", 1, precUnary},
	"nt": {"!", 1, precUnary},
	"nw": {"new", 3, precUnary},
	"nx": {"noexcept", 1, precUnary},
	"oR": {"|=", 2, precAssign},
	"oo": {"||", 2, precLogicalOr},
	"or": {"|", 2, precOr},
	"pL": {"+=", 2, precAssign},
	"pl": {"+", 2, precAdd},
	"pm": {"->*", 2, precPtrMem},
	"pp": {"++", 1, precPostfix},
	"ps": {"+", 1, precUnary},
	"pt": {"->", 2, precPostfix},
	"qu": {"?", 3, precCond},
	"rM": {"%=", 2, precAssign},
	"rS": {">>=", 2, precAssign},
	"rc": {"reinterpret_cast", 2, precPostfix},
	"rm": {"%", 2, precMul},
	"rs": {">>", 2, precShift},
	"sP": {"sizeof...", 1, precUnary},
	"sZ": {"sizeof...", 1, precUnary},
	"sc": {"static_cast", 2, precPostfix},
	"ss": {"<=>", 2, precSpaceship},
	"st": {"sizeof ", 1, precUnary},
	"sz": {"sizeof ", 1, precUnary},
	"te": {"typeid ", 1, precPostfix},
	"ti": {"typeid ", 1, precPostfix},
	"tr": {"throw", 0, precPrimary},
	"tw": {"throw ", 1, precUnary},
}

// operatorName parses:
//
//	operator_name ::= many different two character encodings.
//	              ::= cv <type>
//	              ::= v <digit> <source-name>
//
// We need to know whether we are in an expression because it affects
// how we handle template parameters in the type of a cast operator.
func (st *state) operatorName(inExpression bool) (AST, int) {
	if len(st.str) < 2 {
		st.fail("missing operator code")
	}
	code := st.str[:2]
	st.advance(2)
	if code[0] == 'v' && isDigit(code[1]) {
		name := st.sourceName()
		return &Operator{Name: name.(*Name).Name}, int(code[1] - '0')
	} else if code == "cv" {
		// Push a nil on templates to indicate that template
		// parameters will have their template filled in
		// later.
		if !inExpression {
			st.templates = append(st.templates, nil)
		}

		t := st.demangleType(!inExpression)

		if !inExpression {
			st.templates = st.templates[:len(st.templates)-1]
		}

		return &Cast{To: t}, 1
	} else if op, ok := operators[code]; ok {
		return &Operator{Name: op.name, precedence: op.prec}, op.args
	} else {
		st.failEarlier("unrecognized operator code", 2)
		panic("not reached")
	}
}

// localName parses:
//
//	<local-name> ::= Z <(function) encoding> E <(entity) name> [<discriminator>]
//	             ::= Z <(function) encoding> E s [<discriminator>]
//	             ::= Z <(function) encoding> E d [<parameter> number>] _ <entity name>
//
// Besides the name, this returns whether it saw the code indicating
// a C++23 explicit object parameter.
func (st *state) localName() (AST, bool) {
	st.checkChar('Z')
	fn := st.encoding(true, forLocalName)
	if len(st.str) == 0 || st.str[0] != 'E' {
		st.fail("expected E after local name")
	}
	st.advance(1)
	if len(st.str) > 0 && st.str[0] == 's' {
		st.advance(1)
		var n AST = &Name{Name: "string literal"}
		n = st.discriminator(n)
		return &Qualified{Scope: fn, Name: n, LocalName: true}, false
	} else {
		num := -1
		if len(st.str) > 0 && st.str[0] == 'd' {
			// Default argument scope.
			st.advance(1)
			num = st.compactNumber()
		}
		n, explicitObjectParameter := st.name()
		n = st.discriminator(n)
		if num >= 0 {
			n = &DefaultArg{Num: num, Arg: n}
		}
		return &Qualified{Scope: fn, Name: n, LocalName: true}, explicitObjectParameter
	}
}

// Parse a Java resource special-name.
func (st *state) javaResource() AST {
	off := st.off
	ln := st.number()
	if ln <= 1 {
		st.failEarlier("java resource length less than 1", st.off-off)
	}
	if len(st.str) == 0 || st.str[0] != '_' {
		st.fail("expected _ after number")
	}
	st.advance(1)
	ln--
	if len(st.str) < ln {
		st.fail("not enough characters for java resource length")
	}
	str := st.str[:ln]
	final := ""
	st.advance(ln)
	for i := 0; i < len(str); i++ {
		if str[i] != '$' {
			final += string(str[i])
		} else {
			if len(str) <= i+1 {
				st.failEarlier("java resource escape at end of string", 1)
			}
			i++
			r, ok := map[byte]string{
				'S': "/",
				'_': ".",
				'$': "$",
			}[str[i]]
			if !ok {
				st.failEarlier("unrecognized java resource escape", ln-i-1)
			}
			final += r
		}
	}
	return &Special{Prefix: "java resource ", Val: &Name{Name: final}}
}

// specialName parses:
//
//	<special-name> ::= TV <type>
//	               ::= TT <type>
//	               ::= TI <type>
//	               ::= TS <type>
//	               ::= TA <template-arg>
//	               ::= GV <(object) name>
//	               ::= T <call-offset> <(base) encoding>
//	               ::= Tc <call-offset> <call-offset> <(base) encoding>
//	g++ extensions:
//	               ::= TC <type> <(offset) number> _ <(base) type>
//	               ::= TF <type>
//	               ::= TJ <type>
//	               ::= GR <name>
//	               ::= GA <encoding>
//	               ::= Gr <resource name>
//	               ::= GTt <encoding>
//	               ::= GTn <encoding>
//	               ::= GI <module name>
func (st *state) specialName() AST {
	if st.str[0] == 'T' {
		st.advance(1)
		if len(st.str) == 0 {
			st.fail("expected special name code")
		}
		c := st.str[0]
		st.advance(1)
		switch c {
		case 'V':
			t := st.demangleType(false)
			return &Special{Prefix: "vtable for ", Val: t}
		case 'T':
			t := st.demangleType(false)
			return &Special{Prefix: "VTT for ", Val: t}
		case 'I':
			t := st.demangleType(false)
			return &Special{Prefix: "typeinfo for ", Val: t}
		case 'S':
			t := st.demangleType(false)
			return &Special{Prefix: "typeinfo name for ", Val: t}
		case 'A':
			t := st.templateArg(nil)
			return &Special{Prefix: "template parameter object for ", Val: t}
		case 'h':
			st.callOffset('h')
			v := st.encoding(true, notForLocalName)
			return &Special{Prefix: "non-virtual thunk to ", Val: v}
		case 'v':
			st.callOffset('v')
			v := st.encoding(true, notForLocalName)
			return &Special{Prefix: "virtual thunk to ", Val: v}
		case 'c':
			st.callOffset(0)
			st.callOffset(0)
			v := st.encoding(true, notForLocalName)
			return &Special{Prefix: "covariant return thunk to ", Val: v}
		case 'C':
			derived := st.demangleType(false)
			off := st.off
			offset := st.number()
			if offset < 0 {
				st.failEarlier("expected positive offset", st.off-off)
			}
			if len(st.str) == 0 || st.str[0] != '_' {
				st.fail("expected _ after number")
			}
			st.advance(1)
			base := st.demangleType(false)
			return &Special2{Prefix: "construction vtable for ", Val1: base, Middle: "-in-", Val2: derived}
		case 'F':
			t := st.demangleType(false)
			return &Special{Prefix: "typeinfo fn for ", Val: t}
		case 'J':
			t := st.demangleType(false)
			return &Special{Prefix: "java Class for ", Val: t}
		case 'H':
			n, _ := st.name()
			return &Special{Prefix: "TLS init function for ", Val: n}
		case 'W':
			n, _ := st.name()
			return &Special{Prefix: "TLS wrapper function for ", Val: n}
		default:
			st.fail("unrecognized special T name code")
			panic("not reached")
		}
	} else {
		st.checkChar('G')
		if len(st.str) == 0 {
			st.fail("expected special name code")
		}
		c := st.str[0]
		st.advance(1)
		switch c {
		case 'V':
			n, _ := st.name()
			return &Special{Prefix: "guard variable for ", Val: n}
		case 'R':
			n, _ := st.name()
			st.seqID(true)
			return &Special{Prefix: "reference temporary for ", Val: n}
		case 'A':
			v := st.encoding(true, notForLocalName)
			return &Special{Prefix: "hidden alias for ", Val: v}
		case 'T':
			if len(st.str) == 0 {
				st.fail("expected special GT name code")
			}
			c := st.str[0]
			st.advance(1)
			v := st.encoding(true, notForLocalName)
			switch c {
			case 'n':
				return &Special{Prefix: "non-transaction clone for ", Val: v}
			default:
				// The proposal is that different
				// letters stand for different types
				// of transactional cloning.  Treat
				// them all the same for now.
				fallthrough
			case 't':
				return &Special{Prefix: "transaction clone for ", Val: v}
			}
		case 'r':
			return st.javaResource()
		case 'I':
			module := st.moduleName(nil)
			if module == nil {
				st.fail("expected module after GI")
			}
			return &Special{Prefix: "initializer for module ", Val: module}
		default:
			st.fail("unrecognized special G name code")
			panic("not reached")
		}
	}
}

// callOffset parses:
//
//	<call-offset> ::= h <nv-offset> _
//	              ::= v <v-offset> _
//
//	<nv-offset> ::= <(offset) number>
//
//	<v-offset> ::= <(offset) number> _ <(virtual offset) number>
//
// The c parameter, if not 0, is a character we just read which is the
// start of the <call-offset>.
//
// We don't display the offset information anywhere.
func (st *state) callOffset(c byte) {
	if c == 0 {
		if len(st.str) == 0 {
			st.fail("missing call offset")
		}
		c = st.str[0]
		st.advance(1)
	}
	switch c {
	case 'h':
		st.number()
	case 'v':
		st.number()
		if len(st.str) == 0 || st.str[0] != '_' {
			st.fail("expected _ after number")
		}
		st.advance(1)
		st.number()
	default:
		st.failEarlier("unrecognized call offset code", 1)
	}
	if len(st.str) == 0 || st.str[0] != '_' {
		st.fail("expected _ after call offset")
	}
	st.advance(1)
}

// builtinTypes maps the type letter to the type name.
var builtinTypes = map[byte]string{
	'a': "signed char",
	'b': "bool",
	'c': "char",
	'd': "double",
	'e': "long double",
	'f': "float",
	'g': "__float128",
	'h': "unsigned char",
	'i': "int",
	'j': "unsigned int",
	'l': "long",
	'm': "unsigned long",
	'n': "__int128",
	'o': "unsigned __int128",
	's': "short",
	't': "unsigned short",
	'v': "void",
	'w': "wchar_t",
	'x': "long long",
	'y': "unsigned long long",
	'z': "...",
}

// demangleType parses:
//
//	<type> ::= <builtin-type>
//	       ::= <function-type>
//	       ::= <class-enum-type>
//	       ::= <array-type>
//	       ::= <pointer-to-member-type>
//	       ::= <template-param>
//	       ::= <template-template-param> <template-args>
//	       ::= <substitution>
//	       ::= <CV-qualifiers> <type>
//	       ::= P <type>
//	       ::= R <type>
//	       ::= O <type> (C++0x)
//	       ::= C <type>
//	       ::= G <type>
//	       ::= U <source-name> <type>
//
//	<builtin-type> ::= various one letter codes
//	               ::= u <source-name>
func (st *state) demangleType(isCast bool) AST {
	if len(st.str) == 0 {
		st.fail("expected type")
	}

	addSubst := true

	q := st.cvQualifiers()
	if q != nil {
		if len(st.str) == 0 {
			st.fail("expected type")
		}

		// CV-qualifiers before a function type apply to
		// 'this', so avoid adding the unqualified function
		// type to the substitution list.
		if st.str[0] == 'F' {
			addSubst = false
		}
	}

	var ret AST

	// Use correct substitution for a template parameter.
	var sub AST

	if btype, ok := builtinTypes[st.str[0]]; ok {
		ret = &BuiltinType{Name: btype}
		st.advance(1)
		if q != nil {
			ret = &TypeWithQualifiers{Base: ret, Qualifiers: q}
			st.subs.add(ret)
		}
		return ret
	}
	c := st.str[0]
	switch c {
	case 'u':
		st.advance(1)
		ret = st.sourceName()
		if len(st.str) > 0 && st.str[0] == 'I' {
			st.advance(1)
			base := st.demangleType(false)
			if len(st.str) == 0 || st.str[0] != 'E' {
				st.fail("expected E after transformed type")
			}
			st.advance(1)
			ret = &TransformedType{Name: ret.(*Name).Name, Base: base}
		}
	case 'F':
		ret = st.functionType()
	case 'N', 'W', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		ret, _ = st.name()
	case 'A':
		ret = st.arrayType(isCast)
	case 'M':
		ret = st.pointerToMemberType(isCast)
	case 'T':
		if len(st.str) > 1 && (st.str[1] == 's' || st.str[1] == 'u' || st.str[1] == 'e') {
			c = st.str[1]
			st.advance(2)
			ret, _ = st.name()
			var kind string
			switch c {
			case 's':
				kind = "struct"
			case 'u':
				kind = "union"
			case 'e':
				kind = "enum"
			}
			ret = &ElaboratedType{Kind: kind, Type: ret}
			break
		}

		ret = st.templateParam()
		if len(st.str) > 0 && st.str[0] == 'I' {
			// See the function comment to explain this.
			if !isCast {
				st.subs.add(ret)
				args := st.templateArgs()
				ret = &Template{Name: ret, Args: args}
			} else {
				ret = st.demangleCastTemplateArgs(ret, true)
			}
		}
	case 'S':
		// If this is a special substitution, then it
		// is the start of <class-enum-type>.
		var c2 byte
		if len(st.str) > 1 {
			c2 = st.str[1]
		}
		if isDigit(c2) || c2 == '_' || isUpper(c2) {
			ret = st.substitution(false)
			if _, ok := ret.(*ModuleName); ok {
				ret, _ = st.unqualifiedName(ret)
				st.subs.add(ret)
			}
			if len(st.str) == 0 || st.str[0] != 'I' {
				addSubst = false
			} else {
				// See the function comment to explain this.
				if _, ok := ret.(*TemplateParam); !ok || !isCast {
					args := st.templateArgs()
					ret = &Template{Name: ret, Args: args}
				} else {
					next := st.demangleCastTemplateArgs(ret, false)
					if next == ret {
						addSubst = false
					}
					ret = next
				}
			}
		} else {
			ret, _ = st.name()
			// This substitution is not itself a
			// substitution candidate, unless template
			// arguments were added.
			if ret == subAST[c2] || ret == verboseAST[c2] {
				addSubst = false
			}
		}
	case 'O', 'P', 'R', 'C', 'G':
		st.advance(1)
		t := st.demangleType(isCast)
		switch c {
		case 'O':
			ret = &RvalueReferenceType{Base: t}
		case 'P':
			ret = &PointerType{Base: t}
		case 'R':
			ret = &ReferenceType{Base: t}
		case 'C':
			ret = &ComplexType{Base: t}
		case 'G':
			ret = &ImaginaryType{Base: t}
		}
	case 'U':
		if len(st.str) < 2 {
			st.fail("expected source name or unnamed type")
		}
		switch st.str[1] {
		case 'l':
			ret = st.closureTypeName()
			addSubst = false
		case 't':
			ret = st.unnamedTypeName()
			addSubst = false
		default:
			st.advance(1)
			n := st.sourceName()
			if len(st.str) > 0 && st.str[0] == 'I' {
				args := st.templateArgs()
				n = &Template{Name: n, Args: args}
			}
			t := st.demangleType(isCast)
			ret = &VendorQualifier{Qualifier: n, Type: t}
		}
	case 'D':
		st.advance(1)
		if len(st.str) == 0 {
			st.fail("expected D code for type")
		}
		addSubst = false
		c2 := st.str[0]
		st.advance(1)
		switch c2 {
		case 'T', 't':
			// decltype(expression)
			ret = st.expression()
			if len(st.str) == 0 || st.str[0] != 'E' {
				st.fail("expected E after expression in type")
			}
			st.advance(1)
			ret = &Decltype{Expr: ret}
			addSubst = true

		case 'p':
			t := st.demangleType(isCast)
			pack := st.findArgumentPack(t)
			ret = &PackExpansion{Base: t, Pack: pack}
			addSubst = true

		case 'a':
			ret = &Name{Name: "auto"}
		case 'c':
			ret = &Name{Name: "decltype(auto)"}

		case 'f':
			ret = &BuiltinType{Name: "decimal32"}
		case 'd':
			ret = &BuiltinType{Name: "decimal64"}
		case 'e':
			ret = &BuiltinType{Name: "decimal128"}
		case 'h':
			ret = &BuiltinType{Name: "half"}
		case 'u':
			ret = &BuiltinType{Name: "char8_t"}
		case 's':
			ret = &BuiltinType{Name: "char16_t"}
		case 'i':
			ret = &BuiltinType{Name: "char32_t"}
		case 'n':
			ret = &BuiltinType{Name: "decltype(nullptr)"}

		case 'F':
			accum := false
			bits := 0
			if len(st.str) > 0 && isDigit(st.str[0]) {
				accum = true
				bits = st.number()
			}
			if len(st.str) > 0 && st.str[0] == '_' {
				if bits == 0 {
					st.fail("expected non-zero number of bits")
				}
				st.advance(1)
				ret = &BinaryFP{Bits: bits}
			} else {
				base := st.demangleType(isCast)
				if len(st.str) > 0 && isDigit(st.str[0]) {
					// We don't care about the bits.
					st.number()
				}
				sat := false
				if len(st.str) > 0 {
					if st.str[0] == 's' {
						sat = true
					}
					st.advance(1)
				}
				ret = &FixedType{Base: base, Accum: accum, Sat: sat}
			}

		case 'v':
			ret = st.vectorType(isCast)
			addSubst = true

		case 'B', 'U':
			signed := c2 == 'B'
			var size AST
			if len(st.str) > 0 && isDigit(st.str[0]) {
				bits := st.number()
				size = &Name{Name: fmt.Sprintf("%d", bits)}
			} else {
				size = st.expression()
			}
			if len(st.str) == 0 || st.str[0] != '_' {
				st.fail("expected _ after _BitInt size")
			}
			st.advance(1)
			ret = &BitIntType{Size: size, Signed: signed}

		case 'k':
			constraint, _ := st.name()
			ret = &SuffixType{
				Base:   constraint,
				Suffix: "auto",
			}

		case 'K':
			constraint, _ := st.name()
			ret = &SuffixType{
				Base:   constraint,
				Suffix: "decltype(auto)",
			}

		default:
			st.fail("unrecognized D code in type")
		}

	default:
		st.fail("unrecognized type code")
	}

	if addSubst {
		if sub != nil {
			st.subs.add(sub)
		} else {
			st.subs.add(ret)
		}
	}

	if q != nil {
		if _, ok := ret.(*FunctionType); ok {
			ret = &MethodWithQualifiers{Method: ret, Qualifiers: q, RefQualifier: ""}
		} else if mwq, ok := ret.(*MethodWithQualifiers); ok {
			// Merge adjacent qualifiers.  This case
			// happens with a function with a trailing
			// ref-qualifier.
			mwq.Qualifiers = mergeQualifiers(q, mwq.Qualifiers)
		} else {
			// Merge adjacent qualifiers.  This case
			// happens with multi-dimensional array types.
			if qsub, ok := ret.(*TypeWithQualifiers); ok {
				q = mergeQualifiers(q, qsub.Qualifiers)
				ret = qsub.Base
			}
			ret = &TypeWithQualifiers{Base: ret, Qualifiers: q}
		}
		st.subs.add(ret)
	}

	return ret
}

// demangleCastTemplateArgs is for a rather hideous parse.  When we
// see a template-param followed by a template-args, we need to decide
// whether we have a template-param or a template-template-param.
// Normally it is template-template-param, meaning that we pick up the
// template arguments here.  But, if we are parsing the type for a
// cast operator, then the only way this can be template-template-param
// is if there is another set of template-args immediately after this
// set.  That would look like this:
//
//	<nested-name>
//	-> <template-prefix> <template-args>
//	-> <prefix> <template-unqualified-name> <template-args>
//	-> <unqualified-name> <template-unqualified-name> <template-args>
//	-> <source-name> <template-unqualified-name> <template-args>
//	-> <source-name> <operator-name> <template-args>
//	-> <source-name> cv <type> <template-args>
//	-> <source-name> cv <template-template-param> <template-args> <template-args>
//
// Otherwise, we have this derivation:
//
//	<nested-name>
//	-> <template-prefix> <template-args>
//	-> <prefix> <template-unqualified-name> <template-args>
//	-> <unqualified-name> <template-unqualified-name> <template-args>
//	-> <source-name> <template-unqualified-name> <template-args>
//	-> <source-name> <operator-name> <template-args>
//	-> <source-name> cv <type> <template-args>
//	-> <source-name> cv <template-param> <template-args>
//
// in which the template-args are actually part of the prefix.  For
// the special case where this arises, demangleType is called with
// isCast as true.  This function is then responsible for checking
// whether we see <template-param> <template-args> but there is not
// another following <template-args>.  In that case, we reset the
// parse and just return the <template-param>.
func (st *state) demangleCastTemplateArgs(tp AST, addSubst bool) AST {
	save := st.copy()

	var args []AST
	failed := false
	func() {
		defer func() {
			if r := recover(); r != nil {
				if _, ok := r.(demangleErr); ok {
					failed = true
				} else {
					panic(r)
				}
			}
		}()

		args = st.templateArgs()
	}()

	if !failed && len(st.str) > 0 && st.str[0] == 'I' {
		if addSubst {
			st.subs.add(tp)
		}
		return &Template{Name: tp, Args: args}
	}
	// Reset back to before we started reading the template arguments.
	// They will be read again by st.prefix.
	*st = *save
	return tp
}

// mergeQualifiers merges two qualifier lists into one.
func mergeQualifiers(q1AST, q2AST AST) AST {
	if q1AST == nil {
		return q2AST
	}
	if q2AST == nil {
		return q1AST
	}
	q1 := q1AST.(*Qualifiers)
	m := make(map[string]bool)
	for _, qualAST := range q1.Qualifiers {
		qual := qualAST.(*Qualifier)
		if len(qual.Exprs) == 0 {
			m[qual.Name] = true
		}
	}
	rq := q1.Qualifiers
	for _, qualAST := range q2AST.(*Qualifiers).Qualifiers {
		qual := qualAST.(*Qualifier)
		if len(qual.Exprs) > 0 {
			rq = append(rq, qualAST)
		} else if !m[qual.Name] {
			rq = append(rq, qualAST)
			m[qual.Name] = true
		}
	}
	q1.Qualifiers = rq
	return q1
}

// qualifiers maps from the character used in the mangled name to the
// string to print.
var qualifiers = map[byte]string{
	'r': "restrict",
	'V': "volatile",
	'K': "const",
}

// cvQualifiers parses:
//
//	<CV-qualifiers> ::= [r] [V] [K]
func (st *state) cvQualifiers() AST {
	var q []AST
qualLoop:
	for len(st.str) > 0 {
		if qv, ok := qualifiers[st.str[0]]; ok {
			qual := &Qualifier{Name: qv}
			q = append([]AST{qual}, q...)
			st.advance(1)
		} else if len(st.str) > 1 && st.str[0] == 'D' {
			var qual AST
			switch st.str[1] {
			case 'x':
				qual = &Qualifier{Name: "transaction_safe"}
				st.advance(2)
			case 'o':
				qual = &Qualifier{Name: "noexcept"}
				st.advance(2)
			case 'O':
				st.advance(2)
				expr := st.expression()
				if len(st.str) == 0 || st.str[0] != 'E' {
					st.fail("expected E after computed noexcept expression")
				}
				st.advance(1)
				qual = &Qualifier{Name: "noexcept", Exprs: []AST{expr}}
			case 'w':
				st.advance(2)
				parmlist := st.parmlist(false)
				if len(st.str) == 0 || st.str[0] != 'E' {
					st.fail("expected E after throw parameter list")
				}
				st.advance(1)
				qual = &Qualifier{Name: "throw", Exprs: parmlist}
			default:
				break qualLoop
			}
			q = append([]AST{qual}, q...)
		} else {
			break
		}
	}
	if len(q) == 0 {
		return nil
	}
	return &Qualifiers{Qualifiers: q}
}

// refQualifier parses:
//
//	<ref-qualifier> ::= R
//	                ::= O
func (st *state) refQualifier() string {
	if len(st.str) > 0 {
		switch st.str[0] {
		case 'R':
			st.advance(1)
			return "&"
		case 'O':
			st.advance(1)
			return "&&"
		}
	}
	return ""
}

// parmlist parses:
//
//	<type>+
func (st *state) parmlist(explicitObjectParameter bool) []AST {
	var ret []AST
	for {
		if len(st.str) < 1 {
			break
		}
		if st.str[0] == 'E' || st.str[0] == '.' {
			break
		}
		if (st.str[0] == 'R' || st.str[0] == 'O') && len(st.str) > 1 && st.str[1] == 'E' {
			// This is a function ref-qualifier.
			break
		}
		if st.str[0] == 'Q' {
			// This is a requires clause.
			break
		}
		ptype := st.demangleType(false)

		if len(ret) == 0 && explicitObjectParameter {
			ptype = &ExplicitObjectParameter{Base: ptype}
		}

		ret = append(ret, ptype)
	}

	// There should always be at least one type.  A function that
	// takes no arguments will have a single parameter type
	// "void".
	if len(ret) == 0 {
		st.fail("expected at least one type in type list")
	}

	// Omit a single parameter type void.
	if len(ret) == 1 {
		if bt, ok := ret[0].(*BuiltinType); ok && bt.Name == "void" {
			ret = nil
		}
	}

	return ret
}

// functionType parses:
//
//	<function-type> ::= F [Y] <bare-function-type> [<ref-qualifier>] E
func (st *state) functionType() AST {
	st.checkChar('F')
	if len(st.str) > 0 && st.str[0] == 'Y' {
		// Function has C linkage.  We don't print this.
		st.advance(1)
	}
	ret := st.bareFunctionType(true, false)
	r := st.refQualifier()
	if r != "" {
		ret = &MethodWithQualifiers{Method: ret, Qualifiers: nil, RefQualifier: r}
	}
	if len(st.str) == 0 || st.str[0] != 'E' {
		st.fail("expected E after function type")
	}
	st.advance(1)
	return ret
}

// bareFunctionType parses:
//
//	<bare-function-type> ::= [J]<type>+
func (st *state) bareFunctionType(hasReturnType, explicitObjectParameter bool) AST {
	if len(st.str) > 0 && st.str[0] == 'J' {
		hasReturnType = true
		st.advance(1)
	}
	var returnType AST
	if hasReturnType {
		returnType = st.demangleType(false)
	}
	types := st.parmlist(explicitObjectParameter)
	return &FunctionType{
		Return:       returnType,
		Args:         types,
		ForLocalName: false, // may be set later in encoding
	}
}

// arrayType parses:
//
//	<array-type> ::= A <(positive dimension) number> _ <(element) type>
//	             ::= A [<(dimension) expression>] _ <(element) type>
func (st *state) arrayType(isCast bool) AST {
	st.checkChar('A')

	if len(st.str) == 0 {
		st.fail("missing array dimension")
	}

	var dim AST
	if st.str[0] == '_' {
		dim = &Name{Name: ""}
	} else if isDigit(st.str[0]) {
		i := 1
		for len(st.str) > i && isDigit(st.str[i]) {
			i++
		}
		dim = &Name{Name: st.str[:i]}
		st.advance(i)
	} else {
		dim = st.expression()
	}

	if len(st.str) == 0 || st.str[0] != '_' {
		st.fail("expected _ after dimension")
	}
	st.advance(1)

	t := st.demangleType(isCast)

	arr := &ArrayType{Dimension: dim, Element: t}

	// Qualifiers on the element of an array type go on the whole
	// array type.
	if q, ok := arr.Element.(*TypeWithQualifiers); ok {
		return &TypeWithQualifiers{Base: &ArrayType{Dimension: dim, Element: q.Base}, Qualifiers: q.Qualifiers}
	}

	return arr
}

// vectorType parses:
//
//	<vector-type> ::= Dv <number> _ <type>
//	              ::= Dv _ <expression> _ <type>
func (st *state) vectorType(isCast bool) AST {
	if len(st.str) == 0 {
		st.fail("expected vector dimension")
	}

	var dim AST
	if st.str[0] == '_' {
		st.advance(1)
		dim = st.expression()
	} else {
		num := st.number()
		dim = &Name{Name: fmt.Sprintf("%d", num)}
	}

	if len(st.str) == 0 || st.str[0] != '_' {
		st.fail("expected _ after vector dimension")
	}
	st.advance(1)

	t := st.demangleType(isCast)

	return &VectorType{Dimension: dim, Base: t}
}

// pointerToMemberType parses:
//
//	<pointer-to-member-type> ::= M <(class) type> <(member) type>
func (st *state) pointerToMemberType(isCast bool) AST {
	st.checkChar('M')
	cl := st.demangleType(false)

	// The ABI says, "The type of a non-static member function is
	// considered to be different, for the purposes of
	// substitution, from the type of a namespace-scope or static
	// member function whose type appears similar. The types of
	// two non-static member functions are considered to be
	// different, for the purposes of substitution, if the
	// functions are members of different classes. In other words,
	// for the purposes of substitution, the class of which the
	// function is a member is considered part of the type of
	// function."
	//
	// For a pointer to member function, this call to demangleType
	// will end up adding a (possibly qualified) non-member
	// function type to the substitution table, which is not
	// correct; however, the member function type will never be
	// used in a substitution, so putting the wrong type in the
	// substitution table is harmless.
	mem := st.demangleType(isCast)
	return &PtrMem{Class: cl, Member: mem}
}

// compactNumber parses:
//
//	<non-negative number> _
func (st *state) compactNumber() int {
	if len(st.str) == 0 {
		st.fail("missing index")
	}
	if st.str[0] == '_' {
		st.advance(1)
		return 0
	} else if st.str[0] == 'n' {
		st.fail("unexpected negative number")
	}
	n := st.number()
	if len(st.str) == 0 || st.str[0] != '_' {
		st.fail("missing underscore after number")
	}
	st.advance(1)
	return n + 1
}

// templateParam parses:
//
//	<template-param> ::= T_
//	                 ::= T <(parameter-2 non-negative) number> _
//	                 ::= TL <level-1> __
//	                 ::= TL <level-1> _ <parameter-2 non-negative number> _
//
// When a template parameter is a substitution candidate, any
// reference to that substitution refers to the template parameter
// with the same index in the currently active template, not to
// whatever the template parameter would be expanded to here.  We sort
// this out in substitution and simplify.
func (st *state) templateParam() AST {
	off := st.off
	str := st.str
	st.checkChar('T')

	level := 0
	if len(st.str) > 0 && st.str[0] == 'L' {
		st.advance(1)
		level = st.compactNumber()
	}

	n := st.compactNumber()

	// We don't try to substitute template parameters in a
	// constraint expression.
	if st.parsingConstraint {
		return &Name{Name: str[:st.off-1-off]}
	}

	if level >= len(st.templates) {
		if st.lambdaTemplateLevel > 0 && level == st.lambdaTemplateLevel-1 {
			// Lambda auto params are mangled as template params.
			// See https://gcc.gnu.org/PR78252.
			return &LambdaAuto{Index: n}
		}
		st.failEarlier(fmt.Sprintf("template parameter is not in scope of template (level %d >= %d)", level, len(st.templates)), st.off-off)
	}

	template := st.templates[level]

	if template == nil {
		// We are parsing a cast operator.  If the cast is
		// itself a template, then this is a forward
		// reference.  Fill it in later.
		return &TemplateParam{Index: n, Template: nil}
	}

	if n >= len(template.Args) {
		if st.lambdaTemplateLevel > 0 && level == st.lambdaTemplateLevel-1 {
			// Lambda auto params are mangled as template params.
			// See https://gcc.gnu.org/PR78252.
			return &LambdaAuto{Index: n}
		}
		st.failEarlier(fmt.Sprintf("template index out of range (%d >= %d)", n, len(template.Args)), st.off-off)
	}

	return &TemplateParam{Index: n, Template: template}
}

// setTemplate sets the Template field of any TemplateParam's in a.
// This handles the forward referencing template parameters found in
// cast operators.
func (st *state) setTemplate(a AST, tmpl *Template) {
	seen := make(map[AST]bool)
	a.Traverse(func(a AST) bool {
		switch a := a.(type) {
		case *TemplateParam:
			if a.Template != nil {
				if tmpl != nil {
					st.fail("duplicate template parameters")
				}
				return false
			}
			if tmpl == nil {
				st.fail("cast template parameter not in scope of template")
			}
			if a.Index >= len(tmpl.Args) {
				st.fail(fmt.Sprintf("cast template index out of range (%d >= %d)", a.Index, len(tmpl.Args)))
			}
			a.Template = tmpl
			return false
		case *Closure:
			// There are no template params in closure types.
			// https://gcc.gnu.org/PR78252.
			return false
		default:
			if seen[a] {
				return false
			}
			seen[a] = true
			return true
		}
	})
}

// clearTemplateArgs gives an error for any unset Template field in
// args.  This handles erroneous cases where a cast operator with a
// forward referenced template is in the scope of another cast
// operator.
func (st *state) clearTemplateArgs(args []AST) {
	for _, a := range args {
		st.setTemplate(a, nil)
	}
}

// templateArgs parses:
//
//	<template-args> ::= I <template-arg>+ E
func (st *state) templateArgs() []AST {
	if len(st.str) == 0 || (st.str[0] != 'I' && st.str[0] != 'J') {
		panic("internal error")
	}
	st.advance(1)

	var ret []AST
	for len(st.str) == 0 || st.str[0] != 'E' {
		arg := st.templateArg(ret)
		ret = append(ret, arg)

		if len(st.str) > 0 && st.str[0] == 'Q' {
			// A list of template arguments can have a
			// constraint, but we don't demangle it.
			st.constraintExpr()
			if len(st.str) == 0 || st.str[0] != 'E' {
				st.fail("expected end of template arguments after constraint")
			}
		}
	}
	st.advance(1)
	return ret
}

// templateArg parses:
//
//	<template-arg> ::= <type>
//	               ::= X <expression> E
//	               ::= <expr-primary>
//	               ::= J <template-arg>* E
//	               ::= LZ <encoding> E
//	               ::= <template-param-decl> <template-arg>
func (st *state) templateArg(prev []AST) AST {
	if len(st.str) == 0 {
		st.fail("missing template argument")
	}
	switch st.str[0] {
	case 'X':
		st.advance(1)
		expr := st.expression()
		if len(st.str) == 0 || st.str[0] != 'E' {
			st.fail("missing end of expression")
		}
		st.advance(1)
		return expr

	case 'L':
		return st.exprPrimary()

	case 'I', 'J':
		args := st.templateArgs()
		return &ArgumentPack{Args: args}

	case 'T':
		var arg byte
		if len(st.str) > 1 {
			arg = st.str[1]
		}
		switch arg {
		case 'y', 'n', 't', 'p', 'k':
			off := st.off

			// Apparently template references in the
			// template parameter refer to previous
			// arguments in the same template.
			template := &Template{Args: prev}
			st.templates = append(st.templates, template)

			param, _ := st.templateParamDecl()

			st.templates = st.templates[:len(st.templates)-1]

			if param == nil {
				st.failEarlier("expected template parameter as template argument", st.off-off)
			}
			arg := st.templateArg(nil)
			return &TemplateParamQualifiedArg{Param: param, Arg: arg}
		}
		return st.demangleType(false)

	default:
		return st.demangleType(false)
	}
}

// exprList parses a sequence of expressions up to a terminating character.
func (st *state) exprList(stop byte) AST {
	if len(st.str) > 0 && st.str[0] == stop {
		st.advance(1)
		return &ExprList{Exprs: nil}
	}

	var exprs []AST
	for {
		e := st.expression()
		exprs = append(exprs, e)
		if len(st.str) > 0 && st.str[0] == stop {
			st.advance(1)
			break
		}
	}
	return &ExprList{Exprs: exprs}
}

// expression parses:
//
//	<expression> ::= <(unary) operator-name> <expression>
//	             ::= <(binary) operator-name> <expression> <expression>
//	             ::= <(trinary) operator-name> <expression> <expression> <expression>
//	             ::= pp_ <expression>
//	             ::= mm_ <expression>
//	             ::= cl <expression>+ E
//	             ::= cl <expression>+ E
//	             ::= cv <type> <expression>
//	             ::= cv <type> _ <expression>* E
//	             ::= tl <type> <braced-expression>* E
//	             ::= il <braced-expression>* E
//	             ::= [gs] nw <expression>* _ <type> E
//	             ::= [gs] nw <expression>* _ <type> <initializer>
//	             ::= [gs] na <expression>* _ <type> E
//	             ::= [gs] na <expression>* _ <type> <initializer>
//	             ::= [gs] dl <expression>
//	             ::= [gs] da <expression>
//	             ::= dc <type> <expression>
//	             ::= sc <type> <expression>
//	             ::= cc <type> <expression>
//	             ::= mc <parameter type> <expr> [<offset number>] E
//	             ::= rc <type> <expression>
//	             ::= ti <type>
//	             ::= te <expression>
//	             ::= so <referent type> <expr> [<offset number>] <union-selector>* [p] E
//	             ::= st <type>
//	             ::= sz <expression>
//	             ::= at <type>
//	             ::= az <expression>
//	             ::= nx <expression>
//	             ::= <template-param>
//	             ::= <function-param>
//	             ::= dt <expression> <unresolved-name>
//	             ::= pt <expression> <unresolved-name>
//	             ::= ds <expression> <expression>
//	             ::= sZ <template-param>
//	             ::= sZ <function-param>
//	             ::= sP <template-arg>* E
//	             ::= sp <expression>
//	             ::= fl <binary operator-name> <expression>
//	             ::= fr <binary operator-name> <expression>
//	             ::= fL <binary operator-name> <expression> <expression>
//	             ::= fR <binary operator-name> <expression> <expression>
//	             ::= tw <expression>
//	             ::= tr
//	             ::= u <source-name> <template-arg>* E
//	             ::= <unresolved-name>
//	             ::= <expr-primary>
//
//	<function-param> ::= fp <CV-qualifiers> _
//	                 ::= fp <CV-qualifiers> <number>
//	                 ::= fL <number> p <CV-qualifiers> _
//	                 ::= fL <number> p <CV-qualifiers> <number>
//	                 ::= fpT
//
//	<braced-expression> ::= <expression>
//	                    ::= di <field source-name> <braced-expression>
//	                    ::= dx <index expression> <braced-expression>
//	                    ::= dX <range begin expression> <range end expression> <braced-expression>
func (st *state) expression() AST {
	if len(st.str) == 0 {
		st.fail("expected expression")
	}
	if st.str[0] == 'L' {
		return st.exprPrimary()
	} else if st.str[0] == 'T' {
		return st.templateParam()
	} else if st.str[0] == 's' && len(st.str) > 1 && st.str[1] == 'o' {
		st.advance(2)
		return st.subobject()
	} else if st.str[0] == 's' && len(st.str) > 1 && st.str[1] == 'r' {
		return st.unresolvedName()
	} else if st.str[0] == 's' && len(st.str) > 1 && st.str[1] == 'p' {
		st.advance(2)
		e := st.expression()
		pack := st.findArgumentPack(e)
		return &PackExpansion{Base: e, Pack: pack}
	} else if st.str[0] == 's' && len(st.str) > 1 && st.str[1] == 'Z' {
		st.advance(2)
		off := st.off
		e := st.expression()
		ap := st.findArgumentPack(e)
		if ap == nil {
			st.failEarlier("missing argument pack", st.off-off)
		}
		return &SizeofPack{Pack: ap}
	} else if st.str[0] == 's' && len(st.str) > 1 && st.str[1] == 'P' {
		st.advance(2)
		var args []AST
		for len(st.str) == 0 || st.str[0] != 'E' {
			arg := st.templateArg(nil)
			args = append(args, arg)
		}
		st.advance(1)
		return &SizeofArgs{Args: args}
	} else if st.str[0] == 'f' && len(st.str) > 1 && st.str[1] == 'p' {
		st.advance(2)
		if len(st.str) > 0 && st.str[0] == 'T' {
			st.advance(1)
			return &FunctionParam{Index: 0}
		} else {
			// We can see qualifiers here, but we don't
			// include them in the demangled string.
			st.cvQualifiers()
			index := st.compactNumber()
			return &FunctionParam{Index: index + 1}
		}
	} else if st.str[0] == 'f' && len(st.str) > 2 && st.str[1] == 'L' && isDigit(st.str[2]) {
		st.advance(2)
		// We don't include the scope count in the demangled string.
		st.number()
		if len(st.str) == 0 || st.str[0] != 'p' {
			st.fail("expected p after function parameter scope count")
		}
		st.advance(1)
		// We can see qualifiers here, but we don't include them
		// in the demangled string.
		st.cvQualifiers()
		index := st.compactNumber()
		return &FunctionParam{Index: index + 1}
	} else if st.str[0] == 'm' && len(st.str) > 1 && st.str[1] == 'c' {
		st.advance(2)
		typ := st.demangleType(false)
		expr := st.expression()
		offset := 0
		if len(st.str) > 0 && (st.str[0] == 'n' || isDigit(st.str[0])) {
			offset = st.number()
		}
		if len(st.str) == 0 || st.str[0] != 'E' {
			st.fail("expected E after pointer-to-member conversion")
		}
		st.advance(1)
		return &PtrMemCast{
			Type:   typ,
			Expr:   expr,
			Offset: offset,
		}
	} else if isDigit(st.str[0]) || (st.str[0] == 'o' && len(st.str) > 1 && st.str[1] == 'n') {
		if st.str[0] == 'o' {
			// Skip operator function ID.
			st.advance(2)
		}
		n, _ := st.unqualifiedName(nil)
		if len(st.str) > 0 && st.str[0] == 'I' {
			args := st.templateArgs()
			n = &Template{Name: n, Args: args}
		}
		return n
	} else if (st.str[0] == 'i' || st.str[0] == 't') && len(st.str) > 1 && st.str[1] == 'l' {
		// Brace-enclosed initializer list.
		c := st.str[0]
		st.advance(2)
		var t AST
		if c == 't' {
			t = st.demangleType(false)
		}
		exprs := st.exprList('E')
		return &InitializerList{Type: t, Exprs: exprs}
	} else if st.str[0] == 's' && len(st.str) > 1 && st.str[1] == 't' {
		o, _ := st.operatorName(true)
		t := st.demangleType(false)
		return &Unary{Op: o, Expr: t, Suffix: false, SizeofType: true}
	} else if st.str[0] == 'u' {
		st.advance(1)
		name := st.sourceName()
		// Special case __uuidof followed by type or
		// expression, as used by LLVM.
		if n, ok := name.(*Name); ok && n.Name == "__uuidof" {
			if len(st.str) < 2 {
				st.fail("missing uuidof argument")
			}
			var operand AST
			if st.str[0] == 't' {
				st.advance(1)
				operand = st.demangleType(false)
			} else if st.str[0] == 'z' {
				st.advance(1)
				operand = st.expression()
			}
			if operand != nil {
				return &Binary{
					Op:   &Operator{Name: "()"},
					Left: name,
					Right: &ExprList{
						Exprs: []AST{operand},
					},
				}
			}
		}
		var args []AST
		for {
			if len(st.str) == 0 {
				st.fail("missing argument in vendor extended expressoin")
			}
			if st.str[0] == 'E' {
				st.advance(1)
				break
			}
			arg := st.templateArg(nil)
			args = append(args, arg)
		}
		return &Binary{
			Op:    &Operator{Name: "()"},
			Left:  name,
			Right: &ExprList{Exprs: args},
		}
	} else if st.str[0] == 'r' && len(st.str) > 1 && (st.str[1] == 'q' || st.str[1] == 'Q') {
		return st.requiresExpr()
	} else {
		if len(st.str) < 2 {
			st.fail("missing operator code")
		}
		code := st.str[:2]
		o, args := st.operatorName(true)
		switch args {
		case 0:
			return &Nullary{Op: o}

		case 1:
			suffix := false
			if code == "pp" || code == "mm" {
				if len(st.str) > 0 && st.str[0] == '_' {
					st.advance(1)
				} else {
					suffix = true
				}
			}
			var operand AST
			if _, ok := o.(*Cast); ok && len(st.str) > 0 && st.str[0] == '_' {
				st.advance(1)
				operand = st.exprList('E')
			} else {
				operand = st.expression()
			}
			return &Unary{Op: o, Expr: operand, Suffix: suffix, SizeofType: false}

		case 2:
			var left, right AST
			if code == "sc" || code == "dc" || code == "cc" || code == "rc" {
				left = st.demangleType(false)
			} else if code[0] == 'f' {
				left, _ = st.operatorName(true)
				right = st.expression()
				return &Fold{Left: code[1] == 'l', Op: left, Arg1: right, Arg2: nil}
			} else if code == "di" {
				left, _ = st.unqualifiedName(nil)
			} else {
				left = st.expression()
			}
			if code == "cl" || code == "cp" {
				right = st.exprList('E')
			} else if code == "dt" || code == "pt" {
				if len(st.str) > 0 && st.str[0] == 'L' {
					right = st.exprPrimary()
				} else {
					right = st.unresolvedName()
					if len(st.str) > 0 && st.str[0] == 'I' {
						args := st.templateArgs()
						right = &Template{Name: right, Args: args}
					}
				}
			} else {
				right = st.expression()
			}
			return &Binary{Op: o, Left: left, Right: right}

		case 3:
			if code[0] == 'n' {
				if code[1] != 'w' && code[1] != 'a' {
					panic("internal error")
				}
				place := st.exprList('_')
				if place.(*ExprList).Exprs == nil {
					place = nil
				}
				t := st.demangleType(false)
				var ini AST
				if len(st.str) > 0 && st.str[0] == 'E' {
					st.advance(1)
				} else if len(st.str) > 1 && st.str[0] == 'p' && st.str[1] == 'i' {
					// Parenthesized initializer.
					st.advance(2)
					ini = st.exprList('E')
				} else if len(st.str) > 1 && st.str[0] == 'i' && st.str[1] == 'l' {
					// Initializer list.
					ini = st.expression()
				} else {
					st.fail("unrecognized new initializer")
				}
				return &New{Op: o, Place: place, Type: t, Init: ini}
			} else if code[0] == 'f' {
				first, _ := st.operatorName(true)
				second := st.expression()
				third := st.expression()
				return &Fold{Left: code[1] == 'L', Op: first, Arg1: second, Arg2: third}
			} else {
				first := st.expression()
				second := st.expression()
				third := st.expression()
				return &Trinary{Op: o, First: first, Second: second, Third: third}
			}

		default:
			st.fail(fmt.Sprintf("unsupported number of operator arguments: %d", args))
			panic("not reached")
		}
	}
}

// subobject parses:
//
//	<expression> ::= so <referent type> <expr> [<offset number>] <union-selector>* [p] E
//	<union-selector> ::= _ [<number>]
func (st *state) subobject() AST {
	typ := st.demangleType(false)
	expr := st.expression()
	offset := 0
	if len(st.str) > 0 && (st.str[0] == 'n' || isDigit(st.str[0])) {
		offset = st.number()
	}
	var selectors []int
	for len(st.str) > 0 && st.str[0] == '_' {
		st.advance(1)
		selector := 0
		if len(st.str) > 0 && (st.str[0] == 'n' || isDigit(st.str[0])) {
			selector = st.number()
		}
		selectors = append(selectors, selector)
	}
	pastEnd := false
	if len(st.str) > 0 && st.str[0] == 'p' {
		st.advance(1)
		pastEnd = true
	}
	if len(st.str) == 0 || st.str[0] != 'E' {
		st.fail("expected E after subobject")
	}
	st.advance(1)
	return &Subobject{
		Type:      typ,
		SubExpr:   expr,
		Offset:    offset,
		Selectors: selectors,
		PastEnd:   pastEnd,
	}
}

// unresolvedName parses:
//
//	<unresolved-name> ::= [gs] <base-unresolved-name>
//	                  ::= sr <unresolved-type> <base-unresolved-name>
//	                  ::= srN <unresolved-type> <unresolved-qualifier-level>+ E <base-unresolved-name>
//	                  ::= [gs] sr <unresolved-qualifier-level>+ E <base-unresolved-name>
func (st *state) unresolvedName() AST {
	if len(st.str) >= 2 && st.str[:2] == "gs" {
		st.advance(2)
		n := st.unresolvedName()
		return &Unary{
			Op:         &Operator{Name: "::"},
			Expr:       n,
			Suffix:     false,
			SizeofType: false,
		}
	} else if len(st.str) >= 2 && st.str[:2] == "sr" {
		st.advance(2)
		if len(st.str) == 0 {
			st.fail("expected unresolved type")
		}
		switch st.str[0] {
		case 'T', 'D', 'S':
			t := st.demangleType(false)
			n := st.baseUnresolvedName()
			n = &Qualified{Scope: t, Name: n, LocalName: false}
			if len(st.str) > 0 && st.str[0] == 'I' {
				args := st.templateArgs()
				n = &Template{Name: n, Args: args}
				st.subs.add(n)
			}
			return n
		default:
			var s AST
			if st.str[0] == 'N' {
				st.advance(1)
				s = st.demangleType(false)
			}
			for len(st.str) == 0 || st.str[0] != 'E' {
				// GCC does not seem to follow the ABI here.
				// It can emit type/name without an 'E'.
				if s != nil && len(st.str) > 0 && !isDigit(st.str[0]) {
					if q, ok := s.(*Qualified); ok {
						a := q.Scope
						if t, ok := a.(*Template); ok {
							st.subs.add(t.Name)
							st.subs.add(t)
						} else {
							st.subs.add(a)
						}
						return s
					}
				}
				n := st.sourceName()
				if len(st.str) > 0 && st.str[0] == 'I' {
					st.subs.add(n)
					args := st.templateArgs()
					n = &Template{Name: n, Args: args}
				}
				if s == nil {
					s = n
				} else {
					s = &Qualified{Scope: s, Name: n, LocalName: false}
				}
			}
			if s == nil {
				st.fail("missing scope in unresolved name")
			}
			st.advance(1)
			n := st.baseUnresolvedName()
			return &Qualified{Scope: s, Name: n, LocalName: false}
		}
	} else {
		return st.baseUnresolvedName()
	}
}

// baseUnresolvedName parses:
//
//	<base-unresolved-name> ::= <simple-id>
//	                       ::= on <operator-name>
//	                       ::= on <operator-name> <template-args>
//	                       ::= dn <destructor-name>
//
//	<simple-id> ::= <source-name> [ <template-args> ]
func (st *state) baseUnresolvedName() AST {
	var n AST
	if len(st.str) >= 2 && st.str[:2] == "on" {
		st.advance(2)
		n, _ = st.operatorName(true)
	} else if len(st.str) >= 2 && st.str[:2] == "dn" {
		st.advance(2)
		if len(st.str) > 0 && isDigit(st.str[0]) {
			n = st.sourceName()
		} else {
			n = st.demangleType(false)
		}
		n = &Destructor{Name: n}
	} else if len(st.str) > 0 && isDigit(st.str[0]) {
		n = st.sourceName()
	} else {
		// GCC seems to not follow the ABI here: it can have
		// an operator name without on.
		// See https://gcc.gnu.org/PR70182.
		n, _ = st.operatorName(true)
	}
	if len(st.str) > 0 && st.str[0] == 'I' {
		args := st.templateArgs()
		n = &Template{Name: n, Args: args}
	}
	return n
}

// requiresExpr parses:
//
//	<expression> ::= rQ <bare-function-type> _ <requirement>+ E
//	             ::= rq <requirement>+ E
//	<requirement> ::= X <expression> [N] [R <type-constraint>]
//	              ::= T <type>
//	              ::= Q <constraint-expression>
func (st *state) requiresExpr() AST {
	st.checkChar('r')
	if len(st.str) == 0 || (st.str[0] != 'q' && st.str[0] != 'Q') {
		st.fail("expected q or Q in requires clause in expression")
	}
	kind := st.str[0]
	st.advance(1)

	var params []AST
	if kind == 'Q' {
		for len(st.str) > 0 && st.str[0] != '_' {
			typ := st.demangleType(false)
			params = append(params, typ)
		}
		st.advance(1)
	}

	var requirements []AST
	for len(st.str) > 0 && st.str[0] != 'E' {
		var req AST
		switch st.str[0] {
		case 'X':
			st.advance(1)
			expr := st.expression()
			var noexcept bool
			if len(st.str) > 0 && st.str[0] == 'N' {
				st.advance(1)
				noexcept = true
			}
			var typeReq AST
			if len(st.str) > 0 && st.str[0] == 'R' {
				st.advance(1)
				typeReq, _ = st.name()
			}
			req = &ExprRequirement{
				Expr:     expr,
				Noexcept: noexcept,
				TypeReq:  typeReq,
			}

		case 'T':
			st.advance(1)
			typ := st.demangleType(false)
			req = &TypeRequirement{Type: typ}

		case 'Q':
			st.advance(1)
			// We parse a regular expression rather than a
			// constraint expression.
			expr := st.expression()
			req = &NestedRequirement{Constraint: expr}

		default:
			st.fail("unrecognized requirement code")
		}

		requirements = append(requirements, req)
	}

	if len(st.str) == 0 || st.str[0] != 'E' {
		st.fail("expected E after requirements")
	}
	st.advance(1)

	return &RequiresExpr{
		Params:       params,
		Requirements: requirements,
	}
}

// exprPrimary parses:
//
//	<expr-primary> ::= L <type> <(value) number> E
//	               ::= L <type> <(value) float> E
//	               ::= L <mangled-name> E
func (st *state) exprPrimary() AST {
	st.checkChar('L')
	if len(st.str) == 0 {
		st.fail("expected primary expression")

	}

	// Check for 'Z' here because g++ incorrectly omitted the
	// underscore until -fabi-version=3.
	var ret AST
	if st.str[0] == '_' || st.str[0] == 'Z' {
		if st.str[0] == '_' {
			st.advance(1)
		}
		if len(st.str) == 0 || st.str[0] != 'Z' {
			st.fail("expected mangled name")
		}
		st.advance(1)
		ret = st.encoding(true, notForLocalName)
	} else {
		t := st.demangleType(false)

		isArrayType := func(typ AST) bool {
			if twq, ok := typ.(*TypeWithQualifiers); ok {
				typ = twq.Base
			}
			_, ok := typ.(*ArrayType)
			return ok
		}

		neg := false
		if len(st.str) > 0 && st.str[0] == 'n' {
			neg = true
			st.advance(1)
		}
		if len(st.str) > 0 && st.str[0] == 'E' {
			if bt, ok := t.(*BuiltinType); ok && bt.Name == "decltype(nullptr)" {
				// A nullptr should not have a value.
				// We accept one if present because GCC
				// used to generate one.
				// https://gcc.gnu.org/PR91979.
			} else if cl, ok := t.(*Closure); ok {
				// A closure doesn't have a value.
				st.advance(1)
				return &LambdaExpr{Type: cl}
			} else if isArrayType(t) {
				st.advance(1)
				return &StringLiteral{Type: t}
			} else {
				st.fail("missing literal value")
			}
		}
		i := 0
		for len(st.str) > i && st.str[i] != 'E' {
			i++
		}
		val := st.str[:i]
		st.advance(i)
		ret = &Literal{Type: t, Val: val, Neg: neg}
	}
	if len(st.str) == 0 || st.str[0] != 'E' {
		st.fail("expected E after literal")
	}
	st.advance(1)
	return ret
}

// discriminator parses:
//
//	<discriminator> ::= _ <(non-negative) number> (when number < 10)
//	                    __ <(non-negative) number> _ (when number >= 10)
func (st *state) discriminator(a AST) AST {
	if len(st.str) == 0 || st.str[0] != '_' {
		// clang can generate a discriminator at the end of
		// the string with no underscore.
		for i := 0; i < len(st.str); i++ {
			if !isDigit(st.str[i]) {
				return a
			}
		}
		// Skip the trailing digits.
		st.advance(len(st.str))
		return a
	}
	off := st.off
	st.advance(1)
	trailingUnderscore := false
	if len(st.str) > 0 && st.str[0] == '_' {
		st.advance(1)
		trailingUnderscore = true
	}
	d := st.number()
	if d < 0 {
		st.failEarlier("invalid negative discriminator", st.off-off)
	}
	if trailingUnderscore && d >= 10 {
		if len(st.str) == 0 || st.str[0] != '_' {
			st.fail("expected _ after discriminator >= 10")
		}
		st.advance(1)
	}
	// We don't currently print out the discriminator, so we don't
	// save it.
	return a
}

// closureTypeName parses:
//
//	<closure-type-name> ::= Ul <lambda-sig> E [ <nonnegative number> ] _
//	<lambda-sig> ::= <parameter type>+
func (st *state) closureTypeName() AST {
	st.checkChar('U')
	st.checkChar('l')

	oldLambdaTemplateLevel := st.lambdaTemplateLevel
	st.lambdaTemplateLevel = len(st.templates) + 1

	var templateArgs []AST
	var template *Template
	for len(st.str) > 1 && st.str[0] == 'T' {
		arg, templateVal := st.templateParamDecl()
		if arg == nil {
			break
		}
		templateArgs = append(templateArgs, arg)
		if template == nil {
			template = &Template{
				Name: &Name{Name: "lambda"},
			}
			st.templates = append(st.templates, template)
		}
		template.Args = append(template.Args, templateVal)
	}

	var templateArgsConstraint AST
	if len(st.str) > 0 && st.str[0] == 'Q' {
		templateArgsConstraint = st.constraintExpr()
	}

	types := st.parmlist(false)

	st.lambdaTemplateLevel = oldLambdaTemplateLevel

	if template != nil {
		st.templates = st.templates[:len(st.templates)-1]
	}

	var callConstraint AST
	if len(st.str) > 0 && st.str[0] == 'Q' {
		callConstraint = st.constraintExpr()
	}

	if len(st.str) == 0 || st.str[0] != 'E' {
		st.fail("expected E after closure type name")
	}
	st.advance(1)
	num := st.compactNumber()
	return &Closure{
		TemplateArgs:           templateArgs,
		TemplateArgsConstraint: templateArgsConstraint,
		Types:                  types,
		Num:                    num,
		CallConstraint:         callConstraint,
	}
}

// templateParamDecl parses:
//
//	<template-param-decl> ::= Ty                          # type parameter
//	                      ::= Tn <type>                   # non-type parameter
//	                      ::= Tt <template-param-decl>* E # template parameter
//	                      ::= Tp <template-param-decl>    # parameter pack
//
// Returns the new AST to include in the AST we are building and the
// new AST to add to the list of template parameters.
//
// Returns nil, nil if not looking at a template-param-decl.
func (st *state) templateParamDecl() (AST, AST) {
	if len(st.str) < 2 || st.str[0] != 'T' {
		return nil, nil
	}
	mk := func(prefix string, p *int) AST {
		idx := *p
		(*p)++
		return &TemplateParamName{
			Prefix: prefix,
			Index:  idx,
		}
	}
	switch st.str[1] {
	case 'y':
		st.advance(2)
		name := mk("$T", &st.typeTemplateParamCount)
		tp := &TypeTemplateParam{
			Name: name,
		}
		return tp, name
	case 'k':
		st.advance(2)
		constraint, _ := st.name()
		name := mk("$T", &st.typeTemplateParamCount)
		tp := &ConstrainedTypeTemplateParam{
			Name:       name,
			Constraint: constraint,
		}
		return tp, name
	case 'n':
		st.advance(2)
		name := mk("$N", &st.nonTypeTemplateParamCount)
		typ := st.demangleType(false)
		tp := &NonTypeTemplateParam{
			Name: name,
			Type: typ,
		}
		return tp, name
	case 't':
		st.advance(2)
		name := mk("$TT", &st.templateTemplateParamCount)
		var params []AST
		var template *Template
		var constraint AST
		for {
			if len(st.str) == 0 {
				st.fail("expected closure template parameter")
			}
			if st.str[0] == 'E' {
				st.advance(1)
				break
			}
			off := st.off
			param, templateVal := st.templateParamDecl()
			if param == nil {
				st.failEarlier("expected closure template parameter", st.off-off)
			}
			params = append(params, param)
			if template == nil {
				template = &Template{
					Name: &Name{Name: "template_template"},
				}
				st.templates = append(st.templates, template)
			}
			template.Args = append(template.Args, templateVal)

			if len(st.str) > 0 && st.str[0] == 'Q' {
				// A list of template template
				// parameters can have a constraint.
				constraint = st.constraintExpr()
				if len(st.str) == 0 || st.str[0] != 'E' {
					st.fail("expected end of template template parameters after constraint")
				}
			}
		}
		if template != nil {
			st.templates = st.templates[:len(st.templates)-1]
		}
		tp := &TemplateTemplateParam{
			Name:       name,
			Params:     params,
			Constraint: constraint,
		}
		return tp, name
	case 'p':
		st.advance(2)
		off := st.off
		param, templateVal := st.templateParamDecl()
		if param == nil {
			st.failEarlier("expected lambda template parameter", st.off-off)
		}
		return &TemplateParamPack{Param: param}, templateVal
	default:
		return nil, nil
	}
}

// unnamedTypeName parses:
//
//	<unnamed-type-name> ::= Ut [ <nonnegative number> ] _
func (st *state) unnamedTypeName() AST {
	st.checkChar('U')
	st.checkChar('t')
	num := st.compactNumber()
	ret := &UnnamedType{Num: num}
	st.subs.add(ret)
	return ret
}

// constraintExpr parses a constraint expression. This is just a
// regular expression, but template parameters are handled specially.
func (st *state) constraintExpr() AST {
	st.checkChar('Q')

	hold := st.parsingConstraint
	st.parsingConstraint = true
	defer func() { st.parsingConstraint = hold }()

	return st.expression()
}

// Recognize a clone suffix.  These are not part of the mangling API,
// but are added by GCC when cloning functions.
func (st *state) cloneSuffix(a AST) AST {
	i := 0
	if len(st.str) > 1 && st.str[0] == '.' && (isLower(st.str[1]) || isDigit(st.str[1]) || st.str[1] == '_') {
		i += 2
		for len(st.str) > i && (isLower(st.str[i]) || isDigit(st.str[i]) || st.str[i] == '_') {
			i++
		}
	}
	for len(st.str) > i+1 && st.str[i] == '.' && isDigit(st.str[i+1]) {
		i += 2
		for len(st.str) > i && isDigit(st.str[i]) {
			i++
		}
	}
	suffix := st.str[:i]
	st.advance(i)
	return &Clone{Base: a, Suffix: suffix}
}

// substitutions is the list of substitution candidates that may
// appear later in the string.
type substitutions []AST

// add adds a new substitution candidate.
func (subs *substitutions) add(a AST) {
	*subs = append(*subs, a)
}

// subAST maps standard substitution codes to the corresponding AST.
var subAST = map[byte]AST{
	't': &Name{Name: "std"},
	'a': &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "allocator"}},
	'b': &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "basic_string"}},
	's': &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "string"}},
	'i': &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "istream"}},
	'o': &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "ostream"}},
	'd': &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "iostream"}},
}

// verboseAST maps standard substitution codes to the long form of the
// corresponding AST.  We use this when the Verbose option is used, to
// match the standard demangler.
var verboseAST = map[byte]AST{
	't': &Name{Name: "std"},
	'a': &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "allocator"}},
	'b': &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "basic_string"}},

	// std::basic_string<char, std::char_traits<char>, std::allocator<char> >
	's': &Template{
		Name: &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "basic_string"}},
		Args: []AST{
			&BuiltinType{Name: "char"},
			&Template{
				Name: &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "char_traits"}},
				Args: []AST{&BuiltinType{Name: "char"}}},
			&Template{
				Name: &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "allocator"}},
				Args: []AST{&BuiltinType{Name: "char"}}}}},
	// std::basic_istream<char, std::char_traits<char> >
	'i': &Template{
		Name: &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "basic_istream"}},
		Args: []AST{
			&BuiltinType{Name: "char"},
			&Template{
				Name: &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "char_traits"}},
				Args: []AST{&BuiltinType{Name: "char"}}}}},
	// std::basic_ostream<char, std::char_traits<char> >
	'o': &Template{
		Name: &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "basic_ostream"}},
		Args: []AST{
			&BuiltinType{Name: "char"},
			&Template{
				Name: &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "char_traits"}},
				Args: []AST{&BuiltinType{Name: "char"}}}}},
	// std::basic_iostream<char, std::char_traits<char> >
	'd': &Template{
		Name: &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "basic_iostream"}},
		Args: []AST{
			&BuiltinType{Name: "char"},
			&Template{
				Name: &Qualified{Scope: &Name{Name: "std"}, Name: &Name{Name: "char_traits"}},
				Args: []AST{&BuiltinType{Name: "char"}}}}},
}

// substitution parses:
//
//	<substitution> ::= S <seq-id> _
//	               ::= S_
//	               ::= St
//	               ::= Sa
//	               ::= Sb
//	               ::= Ss
//	               ::= Si
//	               ::= So
//	               ::= Sd
func (st *state) substitution(forPrefix bool) AST {
	st.checkChar('S')
	if len(st.str) == 0 {
		st.fail("missing substitution index")
	}
	c := st.str[0]
	off := st.off
	if c == '_' || isDigit(c) || isUpper(c) {
		id := st.seqID(false)
		if id >= len(st.subs) {
			st.failEarlier(fmt.Sprintf("substitution index out of range (%d >= %d)", id, len(st.subs)), st.off-off)
		}

		ret := st.subs[id]

		// We need to update any references to template
		// parameters to refer to the currently active
		// template.

		// When copying a Typed we may need to adjust
		// the templates.
		copyTemplates := st.templates
		var oldLambdaTemplateLevel []int

		// pushTemplate is called from skip, popTemplate from copy.
		pushTemplate := func(template *Template) {
			copyTemplates = append(copyTemplates, template)
			oldLambdaTemplateLevel = append(oldLambdaTemplateLevel, st.lambdaTemplateLevel)
			st.lambdaTemplateLevel = 0
		}
		popTemplate := func() {
			copyTemplates = copyTemplates[:len(copyTemplates)-1]
			st.lambdaTemplateLevel = oldLambdaTemplateLevel[len(oldLambdaTemplateLevel)-1]
			oldLambdaTemplateLevel = oldLambdaTemplateLevel[:len(oldLambdaTemplateLevel)-1]
		}

		copy := func(a AST) AST {
			var index int
			switch a := a.(type) {
			case *Typed:
				// Remove the template added in skip.
				if _, ok := a.Name.(*Template); ok {
					popTemplate()
				}
				return nil
			case *Closure:
				// Undo the save in skip.
				st.lambdaTemplateLevel = oldLambdaTemplateLevel[len(oldLambdaTemplateLevel)-1]
				oldLambdaTemplateLevel = oldLambdaTemplateLevel[:len(oldLambdaTemplateLevel)-1]
				return nil
			case *TemplateParam:
				index = a.Index
			case *LambdaAuto:
				// A lambda auto parameter is represented
				// as a template parameter, so we may have
				// to change back when substituting.
				index = a.Index
			default:
				return nil
			}
			if st.parsingConstraint {
				// We don't try to substitute template
				// parameters in a constraint expression.
				return &Name{Name: fmt.Sprintf("T%d", index)}
			}
			if st.lambdaTemplateLevel > 0 {
				if _, ok := a.(*LambdaAuto); ok {
					return nil
				}
				return &LambdaAuto{Index: index}
			}
			var template *Template
			if len(copyTemplates) > 0 {
				template = copyTemplates[len(copyTemplates)-1]
			} else if rt, ok := ret.(*Template); ok {
				// At least with clang we can see a template
				// to start, and sometimes we need to refer
				// to it. There is probably something wrong
				// here.
				template = rt
			} else {
				st.failEarlier("substituted template parameter not in scope of template", st.off-off)
			}
			if template == nil {
				// This template parameter is within
				// the scope of a cast operator.
				return &TemplateParam{Index: index, Template: nil}
			}

			if index >= len(template.Args) {
				st.failEarlier(fmt.Sprintf("substituted template index out of range (%d >= %d)", index, len(template.Args)), st.off-off)
			}

			return &TemplateParam{Index: index, Template: template}
		}
		seen := make(map[AST]bool)
		skip := func(a AST) bool {
			switch a := a.(type) {
			case *Typed:
				if template, ok := a.Name.(*Template); ok {
					// This template is removed in copy.
					pushTemplate(template)
				}
				return false
			case *Closure:
				// This is undone in copy.
				oldLambdaTemplateLevel = append(oldLambdaTemplateLevel, st.lambdaTemplateLevel)
				st.lambdaTemplateLevel = len(copyTemplates) + 1
				return false
			case *TemplateParam, *LambdaAuto:
				return false
			}
			if seen[a] {
				return true
			}
			seen[a] = true
			return false
		}

		if c := ret.Copy(copy, skip); c != nil {
			return c
		}

		return ret
	} else {
		st.advance(1)
		m := subAST
		if st.verbose {
			m = verboseAST
		}
		// For compatibility with the standard demangler, use
		// a longer name for a constructor or destructor.
		if forPrefix && len(st.str) > 0 && (st.str[0] == 'C' || st.str[0] == 'D') {
			m = verboseAST
		}
		a, ok := m[c]
		if !ok {
			st.failEarlier("unrecognized substitution code", 1)
		}

		if len(st.str) > 0 && st.str[0] == 'B' {
			a = st.taggedName(a)
			st.subs.add(a)
		}

		return a
	}
}

// isDigit returns whetner c is a digit for demangling purposes.
func isDigit(c byte) bool {
	return c >= '0' && c <= '9'
}

// isUpper returns whether c is an upper case letter for demangling purposes.
func isUpper(c byte) bool {
	return c >= 'A' && c <= 'Z'
}

// isLower returns whether c is a lower case letter for demangling purposes.
func isLower(c byte) bool {
	return c >= 'a' && c <= 'z'
}

// simplify replaces template parameters with their expansions, and
// merges qualifiers.
func simplify(a AST) AST {
	seen := make(map[AST]bool)
	skip := func(a AST) bool {
		if seen[a] {
			return true
		}
		seen[a] = true
		return false
	}
	if r := a.Copy(simplifyOne, skip); r != nil {
		return r
	}
	return a
}

// simplifyOne simplifies a single AST.  It returns nil if there is
// nothing to do.
func simplifyOne(a AST) AST {
	switch a := a.(type) {
	case *TemplateParam:
		if a.Template != nil && a.Index < len(a.Template.Args) {
			return a.Template.Args[a.Index]
		}
	case *MethodWithQualifiers:
		if m, ok := a.Method.(*MethodWithQualifiers); ok {
			ref := a.RefQualifier
			if ref == "" {
				ref = m.RefQualifier
			} else if m.RefQualifier != "" {
				if ref == "&" || m.RefQualifier == "&" {
					ref = "&"
				}
			}
			return &MethodWithQualifiers{Method: m.Method, Qualifiers: mergeQualifiers(a.Qualifiers, m.Qualifiers), RefQualifier: ref}
		}
		if t, ok := a.Method.(*TypeWithQualifiers); ok {
			return &MethodWithQualifiers{Method: t.Base, Qualifiers: mergeQualifiers(a.Qualifiers, t.Qualifiers), RefQualifier: a.RefQualifier}
		}
	case *TypeWithQualifiers:
		if ft, ok := a.Base.(*FunctionType); ok {
			return &MethodWithQualifiers{Method: ft, Qualifiers: a.Qualifiers, RefQualifier: ""}
		}
		if t, ok := a.Base.(*TypeWithQualifiers); ok {
			return &TypeWithQualifiers{Base: t.Base, Qualifiers: mergeQualifiers(a.Qualifiers, t.Qualifiers)}
		}
		if m, ok := a.Base.(*MethodWithQualifiers); ok {
			return &MethodWithQualifiers{Method: m.Method, Qualifiers: mergeQualifiers(a.Qualifiers, m.Qualifiers), RefQualifier: m.RefQualifier}
		}
	case *ReferenceType:
		if rt, ok := a.Base.(*ReferenceType); ok {
			return rt
		}
		if rrt, ok := a.Base.(*RvalueReferenceType); ok {
			return &ReferenceType{Base: rrt.Base}
		}
	case *RvalueReferenceType:
		if rrt, ok := a.Base.(*RvalueReferenceType); ok {
			return rrt
		}
		if rt, ok := a.Base.(*ReferenceType); ok {
			return rt
		}
	case *ArrayType:
		// Qualifiers on the element of an array type
		// go on the whole array type.
		if q, ok := a.Element.(*TypeWithQualifiers); ok {
			return &TypeWithQualifiers{
				Base:       &ArrayType{Dimension: a.Dimension, Element: q.Base},
				Qualifiers: q.Qualifiers,
			}
		}
	case *PackExpansion:
		// Expand the pack and replace it with a list of
		// expressions.
		if a.Pack != nil {
			exprs := make([]AST, len(a.Pack.Args))
			for i, arg := range a.Pack.Args {
				copy := func(sub AST) AST {
					// Replace the ArgumentPack
					// with a specific argument.
					if sub == a.Pack {
						return arg
					}
					// Copy everything else.
					return nil
				}

				seen := make(map[AST]bool)
				skip := func(sub AST) bool {
					// Don't traverse into another
					// pack expansion.
					if _, ok := sub.(*PackExpansion); ok {
						return true
					}
					if seen[sub] {
						return true
					}
					seen[sub] = true
					return false
				}

				b := a.Base.Copy(copy, skip)
				if b == nil {
					b = a.Base
				}
				exprs[i] = simplify(b)
			}
			return &ExprList{Exprs: exprs}
		}
	}
	return nil
}

// findArgumentPack walks the AST looking for the argument pack for a
// pack expansion.  We find it via a template parameter.
func (st *state) findArgumentPack(a AST) *ArgumentPack {
	seen := make(map[AST]bool)
	var ret *ArgumentPack
	a.Traverse(func(a AST) bool {
		if ret != nil {
			return false
		}
		switch a := a.(type) {
		case *TemplateParam:
			if a.Template == nil || a.Index >= len(a.Template.Args) {
				return true
			}
			if pack, ok := a.Template.Args[a.Index].(*ArgumentPack); ok {
				ret = pack
				return false
			}
		case *PackExpansion, *Closure, *Name:
			return false
		case *TaggedName, *Operator, *BuiltinType, *FunctionParam:
			return false
		case *UnnamedType, *FixedType, *DefaultArg:
			return false
		}
		if seen[a] {
			return false
		}
		seen[a] = true
		return true
	})
	return ret
}
