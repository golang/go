// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package demangle defines functions that demangle GCC/LLVM C++ symbol names.
// This package recognizes names that were mangled according to the C++ ABI
// defined at http://codesourcery.com/cxx-abi/.
package demangle

import (
	"errors"
	"fmt"
	"strings"
)

// ErrNotMangledName is returned by CheckedDemangle if the string does
// not appear to be a C++ symbol name.
var ErrNotMangledName = errors.New("not a C++ mangled name")

// Option is the type of demangler options.
type Option int

const (
	// The NoParams option disables demangling of function parameters.
	NoParams Option = iota

	// The NoTemplateParams option disables demangling of template parameters.
	NoTemplateParams

	// The NoClones option disables inclusion of clone suffixes.
	// NoParams implies NoClones.
	NoClones

	// The Verbose option turns on more verbose demangling.
	Verbose
)

// Filter demangles a C++ symbol name, returning the human-readable C++ name.
// If any error occurs during demangling, the input string is returned.
func Filter(name string, options ...Option) string {
	ret, err := ToString(name, options...)
	if err != nil {
		return name
	}
	return ret
}

// ToString demangles a C++ symbol name, returning human-readable C++
// name or an error.
// If the name does not appear to be a C++ symbol name at all, the
// error will be ErrNotMangledName.
func ToString(name string, options ...Option) (string, error) {
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
func ToAST(name string, options ...Option) (AST, error) {
	if strings.HasPrefix(name, "_Z") {
		a, err := doDemangle(name[2:], options...)
		return a, adjustErr(err, 2)
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
		switch o {
		case NoParams:
			params = false
			clones = false
		case NoTemplateParams:
		// This is a valid option but only affect printing of the AST.
		case NoClones:
			clones = false
		case Verbose:
			verbose = true
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

// encoding ::= <(function) name> <bare-function-type>
//              <(data) name>
//              <special-name>
func (st *state) encoding(params bool, local forLocalNameType) AST {
	if len(st.str) < 1 {
		st.fail("expected encoding")
	}

	if st.str[0] == 'G' || st.str[0] == 'T' {
		return st.specialName()
	}

	a := st.name()
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

	check := a
	mwq, _ := check.(*MethodWithQualifiers)
	if mwq != nil {
		check = mwq.Method
	}
	template, _ := check.(*Template)
	if template != nil {
		st.templates = append(st.templates, template)
	}

	ft := st.bareFunctionType(hasReturnType(a))

	if template != nil {
		st.templates = st.templates[:len(st.templates)-1]
	}

	ft = simplify(ft)

	// For a local name, discard the return type, so that it
	// doesn't get confused with the top level return type.
	if local == forLocalName {
		if functype, ok := ft.(*FunctionType); ok {
			functype.Return = nil
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

	return &Typed{Name: a, Type: ft}
}

// hasReturnType returns whether the mangled form of a will have a
// return type.
func hasReturnType(a AST) bool {
	switch a := a.(type) {
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

// <tagged-name> ::= <name> B <source-name>
func (st *state) taggedName(a AST) AST {
	for len(st.str) > 0 && st.str[0] == 'B' {
		st.advance(1)
		tag := st.sourceName()
		a = &TaggedName{Name: a, Tag: tag}
	}
	return a
}

// <name> ::= <nested-name>
//        ::= <unscoped-name>
//        ::= <unscoped-template-name> <template-args>
//        ::= <local-name>
//
// <unscoped-name> ::= <unqualified-name>
//                 ::= St <unqualified-name>
//
// <unscoped-template-name> ::= <unscoped-name>
//                          ::= <substitution>
func (st *state) name() AST {
	if len(st.str) < 1 {
		st.fail("expected name")
	}
	switch st.str[0] {
	case 'N':
		return st.nestedName()
	case 'Z':
		return st.localName()
	case 'U':
		a, isCast := st.unqualifiedName()
		if isCast {
			st.setTemplate(a, nil)
		}
		return a
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
			a, isCast = st.unqualifiedName()
			a = &Qualified{Scope: &Name{Name: "std"}, Name: a, LocalName: false}
		} else {
			a = st.substitution(false)
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
		return a

	default:
		a, isCast := st.unqualifiedName()
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
		return a
	}
}

// <nested-name> ::= N [<CV-qualifiers>] [<ref-qualifier>] <prefix> <unqualified-name> E
//               ::= N [<CV-qualifiers>] [<ref-qualifier>] <template-prefix> <template-args> E
func (st *state) nestedName() AST {
	st.checkChar('N')
	q := st.cvQualifiers()
	r := st.refQualifier()
	a := st.prefix()
	if len(q) > 0 || r != "" {
		a = &MethodWithQualifiers{Method: a, Qualifiers: q, RefQualifier: r}
	}
	if len(st.str) == 0 || st.str[0] != 'E' {
		st.fail("expected E after nested name")
	}
	st.advance(1)
	return a
}

// <prefix> ::= <prefix> <unqualified-name>
//          ::= <template-prefix> <template-args>
//          ::= <template-param>
//          ::= <decltype>
//          ::=
//          ::= <substitution>
//
// <template-prefix> ::= <prefix> <(template) unqualified-name>
//                   ::= <template-param>
//                   ::= <substitution>
//
// <decltype> ::= Dt <expression> E
//            ::= DT <expression> E
func (st *state) prefix() AST {
	var a AST

	// The last name seen, for a constructor/destructor.
	var last AST

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

	isCast := false
	for {
		if len(st.str) == 0 {
			st.fail("expected prefix")
		}
		var next AST

		c := st.str[0]
		if isDigit(c) || isLower(c) || c == 'U' || c == 'L' {
			un, isUnCast := st.unqualifiedName()
			next = un
			if isUnCast {
				isCast = true
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
				if inheriting {
					last = st.demangleType(false)
				}
				next = &Constructor{Name: getLast(last)}
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
				}
			case 'S':
				next = st.substitution(true)
			case 'I':
				if a == nil {
					st.fail("unexpected template arguments")
				}
				var args []AST
				args = st.templateArgs()
				tmpl := &Template{Name: a, Args: args}
				if isCast {
					st.setTemplate(a, tmpl)
					st.clearTemplateArgs(args)
					isCast = false
				}
				a = nil
				next = tmpl
			case 'T':
				next = st.templateParam()
			case 'E':
				if a == nil {
					st.fail("expected prefix")
				}
				if isCast {
					st.setTemplate(a, nil)
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
			default:
				st.fail("unrecognized letter in prefix")
			}
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

// <unqualified-name> ::= <operator-name>
//                    ::= <ctor-dtor-name>
//                    ::= <source-name>
//                    ::= <local-source-name>
//
//  <local-source-name>	::= L <source-name> <discriminator>
func (st *state) unqualifiedName() (r AST, isCast bool) {
	if len(st.str) < 1 {
		st.fail("expected unqualified name")
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

	if len(st.str) > 0 && st.str[0] == 'B' {
		a = st.taggedName(a)
	}

	return a, isCast
}

// <source-name> ::= <(positive length) number> <identifier>
// identifier ::= <(unqualified source code identifier)>
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

// number ::= [n] <(non-negative decimal integer)>
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

// An operator is the demangled name, and the number of arguments it
// takes in an expression.
type operator struct {
	name string
	args int
}

// The operators map maps the mangled operator names to information
// about them.
var operators = map[string]operator{
	"aN": {"&=", 2},
	"aS": {"=", 2},
	"aa": {"&&", 2},
	"ad": {"&", 1},
	"an": {"&", 2},
	"at": {"alignof ", 1},
	"az": {"alignof ", 1},
	"cc": {"const_cast", 2},
	"cl": {"()", 2},
	"cm": {",", 2},
	"co": {"~", 1},
	"dV": {"/=", 2},
	"da": {"delete[] ", 1},
	"dc": {"dynamic_cast", 2},
	"de": {"*", 1},
	"dl": {"delete ", 1},
	"ds": {".*", 2},
	"dt": {".", 2},
	"dv": {"/", 2},
	"eO": {"^=", 2},
	"eo": {"^", 2},
	"eq": {"==", 2},
	"fl": {"...", 2},
	"fr": {"...", 2},
	"fL": {"...", 3},
	"fR": {"...", 3},
	"ge": {">=", 2},
	"gs": {"::", 1},
	"gt": {">", 2},
	"ix": {"[]", 2},
	"lS": {"<<=", 2},
	"le": {"<=", 2},
	"li": {`operator"" `, 1},
	"ls": {"<<", 2},
	"lt": {"<", 2},
	"mI": {"-=", 2},
	"mL": {"*=", 2},
	"mi": {"-", 2},
	"ml": {"*", 2},
	"mm": {"--", 1},
	"na": {"new[]", 3},
	"ne": {"!=", 2},
	"ng": {"-", 1},
	"nt": {"!", 1},
	"nw": {"new", 3},
	"oR": {"|=", 2},
	"oo": {"||", 2},
	"or": {"|", 2},
	"pL": {"+=", 2},
	"pl": {"+", 2},
	"pm": {"->*", 2},
	"pp": {"++", 1},
	"ps": {"+", 1},
	"pt": {"->", 2},
	"qu": {"?", 3},
	"rM": {"%=", 2},
	"rS": {">>=", 2},
	"rc": {"reinterpret_cast", 2},
	"rm": {"%", 2},
	"rs": {">>", 2},
	"sc": {"static_cast", 2},
	"st": {"sizeof ", 1},
	"sz": {"sizeof ", 1},
	"tr": {"throw", 0},
	"tw": {"throw ", 1},
}

// operator_name ::= many different two character encodings.
//               ::= cv <type>
//               ::= v <digit> <source-name>
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
		return &Operator{Name: op.name}, op.args
	} else {
		st.failEarlier("unrecognized operator code", 2)
		panic("not reached")
	}
}

// <local-name> ::= Z <(function) encoding> E <(entity) name> [<discriminator>]
//              ::= Z <(function) encoding> E s [<discriminator>]
//              ::= Z <(function) encoding> E d [<parameter> number>] _ <entity name>
func (st *state) localName() AST {
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
		return &Qualified{Scope: fn, Name: n, LocalName: true}
	} else {
		num := -1
		if len(st.str) > 0 && st.str[0] == 'd' {
			// Default argument scope.
			st.advance(1)
			num = st.compactNumber()
		}
		n := st.name()
		n = st.discriminator(n)
		if num >= 0 {
			n = &DefaultArg{Num: num, Arg: n}
		}
		return &Qualified{Scope: fn, Name: n, LocalName: true}
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

// <special-name> ::= TV <type>
//                ::= TT <type>
//                ::= TI <type>
//                ::= TS <type>
//                ::= GV <(object) name>
//                ::= T <call-offset> <(base) encoding>
//                ::= Tc <call-offset> <call-offset> <(base) encoding>
// Also g++ extensions:
//                ::= TC <type> <(offset) number> _ <(base) type>
//                ::= TF <type>
//                ::= TJ <type>
//                ::= GR <name>
//                ::= GA <encoding>
//                ::= Gr <resource name>
//                ::= GTt <encoding>
//                ::= GTn <encoding>
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
			n := st.name()
			return &Special{Prefix: "TLS init function for ", Val: n}
		case 'W':
			n := st.name()
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
			n := st.name()
			return &Special{Prefix: "guard variable for ", Val: n}
		case 'R':
			n := st.name()
			i := st.number()
			return &Special{Prefix: fmt.Sprintf("reference temporary #%d for ", i), Val: n}
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
		default:
			st.fail("unrecognized special G name code")
			panic("not reached")
		}
	}
}

// <call-offset> ::= h <nv-offset> _
//               ::= v <v-offset> _
//
// <nv-offset> ::= <(offset) number>
//
// <v-offset> ::= <(offset) number> _ <(virtual offset) number>
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

// <type> ::= <builtin-type>
//        ::= <function-type>
//        ::= <class-enum-type>
//        ::= <array-type>
//        ::= <pointer-to-member-type>
//        ::= <template-param>
//        ::= <template-template-param> <template-args>
//        ::= <substitution>
//        ::= <CV-qualifiers> <type>
//        ::= P <type>
//        ::= R <type>
//        ::= O <type> (C++0x)
//        ::= C <type>
//        ::= G <type>
//        ::= U <source-name> <type>
//
// <builtin-type> ::= various one letter codes
//                ::= u <source-name>
func (st *state) demangleType(isCast bool) AST {
	if len(st.str) == 0 {
		st.fail("expected type")
	}

	addSubst := true

	q := st.cvQualifiers()
	if len(q) > 0 {
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
		if len(q) > 0 {
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
	case 'F':
		ret = st.functionType()
	case 'N', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		ret = st.name()
	case 'A':
		ret = st.arrayType(isCast)
	case 'M':
		ret = st.pointerToMemberType(isCast)
	case 'T':
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
			ret = st.name()
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

		case 'f':
			ret = &BuiltinType{Name: "decimal32"}
		case 'd':
			ret = &BuiltinType{Name: "decimal64"}
		case 'e':
			ret = &BuiltinType{Name: "decimal128"}
		case 'h':
			ret = &BuiltinType{Name: "half"}
		case 's':
			ret = &BuiltinType{Name: "char16_t"}
		case 'i':
			ret = &BuiltinType{Name: "char32_t"}
		case 'n':
			ret = &BuiltinType{Name: "decltype(nullptr)"}

		case 'F':
			accum := false
			if len(st.str) > 0 && isDigit(st.str[0]) {
				accum = true
				// We don't care about the bits.
				_ = st.number()
			}
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

		case 'v':
			ret = st.vectorType(isCast)
			addSubst = true

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

	if len(q) > 0 {
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
// <nested-name>
// -> <template-prefix> <template-args>
// -> <prefix> <template-unqualified-name> <template-args>
// -> <unqualified-name> <template-unqualified-name> <template-args>
// -> <source-name> <template-unqualified-name> <template-args>
// -> <source-name> <operator-name> <template-args>
// -> <source-name> cv <type> <template-args>
// -> <source-name> cv <template-template-param> <template-args> <template-args>
//
// Otherwise, we have this derivation:
//
// <nested-name>
// -> <template-prefix> <template-args>
// -> <prefix> <template-unqualified-name> <template-args>
// -> <unqualified-name> <template-unqualified-name> <template-args>
// -> <source-name> <template-unqualified-name> <template-args>
// -> <source-name> <operator-name> <template-args>
// -> <source-name> cv <type> <template-args>
// -> <source-name> cv <template-param> <template-args>
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

// mergeQualifiers merges two qualifer lists into one.
func mergeQualifiers(q1, q2 Qualifiers) Qualifiers {
	m := make(map[string]bool)
	for _, qual := range q1 {
		m[qual] = true
	}
	for _, qual := range q2 {
		if !m[qual] {
			q1 = append(q1, qual)
			m[qual] = true
		}
	}
	return q1
}

// qualifiers maps from the character used in the mangled name to the
// string to print.
var qualifiers = map[byte]string{
	'r': "restrict",
	'V': "volatile",
	'K': "const",
}

// <CV-qualifiers> ::= [r] [V] [K]
func (st *state) cvQualifiers() Qualifiers {
	var q Qualifiers
	for len(st.str) > 0 {
		if qv, ok := qualifiers[st.str[0]]; ok {
			q = append([]string{qv}, q...)
			st.advance(1)
		} else if len(st.str) > 1 && st.str[:2] == "Dx" {
			q = append([]string{"transaction_safe"}, q...)
			st.advance(2)
		} else {
			break
		}
	}
	return q
}

// <ref-qualifier> ::= R
//                 ::= O
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

// <type>+
func (st *state) parmlist() []AST {
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
		ptype := st.demangleType(false)
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

// <function-type> ::= F [Y] <bare-function-type> [<ref-qualifier>] E
func (st *state) functionType() AST {
	st.checkChar('F')
	if len(st.str) > 0 && st.str[0] == 'Y' {
		// Function has C linkage.  We don't print this.
		st.advance(1)
	}
	ret := st.bareFunctionType(true)
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

// <bare-function-type> ::= [J]<type>+
func (st *state) bareFunctionType(hasReturnType bool) AST {
	if len(st.str) > 0 && st.str[0] == 'J' {
		hasReturnType = true
		st.advance(1)
	}
	var returnType AST
	if hasReturnType {
		returnType = st.demangleType(false)
	}
	types := st.parmlist()
	return &FunctionType{Return: returnType, Args: types}
}

// <array-type> ::= A <(positive dimension) number> _ <(element) type>
//              ::= A [<(dimension) expression>] _ <(element) type>
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

// <vector-type> ::= Dv <number> _ <type>
//               ::= Dv _ <expression> _ <type>
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

// <pointer-to-member-type> ::= M <(class) type> <(member) type>
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

// <non-negative number> _ */
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

// <template-param> ::= T_
//                  ::= T <(parameter-2 non-negative) number> _
//
// When a template parameter is a substitution candidate, any
// reference to that substitution refers to the template parameter
// with the same index in the currently active template, not to
// whatever the template parameter would be expanded to here.  We sort
// this out in substitution and simplify.
func (st *state) templateParam() AST {
	if len(st.templates) == 0 {
		st.fail("template parameter not in scope of template")
	}
	off := st.off

	st.checkChar('T')
	n := st.compactNumber()

	template := st.templates[len(st.templates)-1]

	if template == nil {
		// We are parsing a cast operator.  If the cast is
		// itself a template, then this is a forward
		// reference.  Fill it in later.
		return &TemplateParam{Index: n, Template: nil}
	}

	if n >= len(template.Args) {
		st.failEarlier(fmt.Sprintf("template index out of range (%d >= %d)", n, len(template.Args)), st.off-off)
	}

	return &TemplateParam{Index: n, Template: template}
}

// setTemplate sets the Template field of any TemplateParam's in a.
// This handles the forward referencing template parameters found in
// cast operators.
func (st *state) setTemplate(a AST, tmpl *Template) {
	var seen []AST
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
		default:
			for _, v := range seen {
				if v == a {
					return false
				}
			}
			seen = append(seen, a)
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

// <template-args> ::= I <template-arg>+ E
func (st *state) templateArgs() []AST {
	if len(st.str) == 0 || (st.str[0] != 'I' && st.str[0] != 'J') {
		panic("internal error")
	}
	st.advance(1)

	var ret []AST
	for len(st.str) == 0 || st.str[0] != 'E' {
		arg := st.templateArg()
		ret = append(ret, arg)
	}
	st.advance(1)
	return ret
}

// <template-arg> ::= <type>
//                ::= X <expression> E
//                ::= <expr-primary>
func (st *state) templateArg() AST {
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

// <expression> ::= <(unary) operator-name> <expression>
//              ::= <(binary) operator-name> <expression> <expression>
//              ::= <(trinary) operator-name> <expression> <expression> <expression>
//              ::= cl <expression>+ E
//              ::= st <type>
//              ::= <template-param>
//              ::= sr <type> <unqualified-name>
//              ::= sr <type> <unqualified-name> <template-args>
//              ::= <expr-primary>
func (st *state) expression() AST {
	if len(st.str) == 0 {
		st.fail("expected expression")
	}
	if st.str[0] == 'L' {
		return st.exprPrimary()
	} else if st.str[0] == 'T' {
		return st.templateParam()
	} else if st.str[0] == 's' && len(st.str) > 1 && st.str[1] == 'r' {
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
				st.subs.add(s)
			}
			if s == nil {
				st.fail("missing scope in unresolved name")
			}
			st.advance(1)
			n := st.baseUnresolvedName()
			return &Qualified{Scope: s, Name: n, LocalName: false}
		}
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
			arg := st.templateArg()
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
			index := st.compactNumber()
			return &FunctionParam{Index: index + 1}
		}
	} else if isDigit(st.str[0]) || (st.str[0] == 'o' && len(st.str) > 1 && st.str[1] == 'n') {
		if st.str[0] == 'o' {
			// Skip operator function ID.
			st.advance(2)
		}
		n, _ := st.unqualifiedName()
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
			} else {
				left = st.expression()
			}
			if code == "cl" {
				right = st.exprList('E')
			} else if code == "dt" || code == "pt" {
				right, _ = st.unqualifiedName()
				if len(st.str) > 0 && st.str[0] == 'I' {
					args := st.templateArgs()
					right = &Template{Name: right, Args: args}
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

// <base-unresolved-name> ::= <simple-id>
//                        ::= on <operator-name>
//                        ::= on <operator-name> <template-args>
//                        ::= dn <destructor-name>
//
//<simple-id> ::= <source-name> [ <template-args> ]
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

// <expr-primary> ::= L <type> <(value) number> E
//                ::= L <type> <(value) float> E
//                ::= L <mangled-name> E
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

		neg := false
		if len(st.str) > 0 && st.str[0] == 'n' {
			neg = true
			st.advance(1)
		}
		if len(st.str) > 0 && st.str[0] == 'E' {
			st.fail("missing literal value")
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

// <discriminator> ::= _ <(non-negative) number>
func (st *state) discriminator(a AST) AST {
	if len(st.str) == 0 || st.str[0] != '_' {
		return a
	}
	off := st.off
	st.advance(1)
	d := st.number()
	if d < 0 {
		st.failEarlier("invalid negative discriminator", st.off-off)
	}
	// We don't currently print out the discriminator, so we don't
	// save it.
	return a
}

// <closure-type-name> ::= Ul <lambda-sig> E [ <nonnegative number> ] _
func (st *state) closureTypeName() AST {
	st.checkChar('U')
	st.checkChar('l')
	types := st.parmlist()
	if len(st.str) == 0 || st.str[0] != 'E' {
		st.fail("expected E after closure type name")
	}
	st.advance(1)
	num := st.compactNumber()
	ret := &Closure{Types: types, Num: num}
	st.subs.add(ret)
	return ret
}

// <unnamed-type-name> ::= Ut [ <nonnegative number> ] _
func (st *state) unnamedTypeName() AST {
	st.checkChar('U')
	st.checkChar('t')
	num := st.compactNumber()
	ret := &UnnamedType{Num: num}
	st.subs.add(ret)
	return ret
}

// Recognize a clone suffix.  These are not part of the mangling API,
// but are added by GCC when cloning functions.
func (st *state) cloneSuffix(a AST) AST {
	i := 0
	if len(st.str) > 1 && st.str[0] == '.' && (isLower(st.str[1]) || st.str[1] == '_') {
		i += 2
		for len(st.str) > i && (isLower(st.str[i]) || st.str[i] == '_') {
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

// <substitution> ::= S <seq-id> _
//                ::= S_
//                ::= St
//                ::= Sa
//                ::= Sb
//                ::= Ss
//                ::= Si
//                ::= So
//                ::= Sd
func (st *state) substitution(forPrefix bool) AST {
	st.checkChar('S')
	if len(st.str) == 0 {
		st.fail("missing substitution index")
	}
	c := st.str[0]
	st.advance(1)
	dec := 1
	if c == '_' || isDigit(c) || isUpper(c) {
		id := 0
		if c != '_' {
			for c != '_' {
				// Don't overflow a 32-bit int.
				if id >= 0x80000000/36-36 {
					st.fail("substitution index overflow")
				}
				if isDigit(c) {
					id = id*36 + int(c-'0')
				} else if isUpper(c) {
					id = id*36 + int(c-'A') + 10
				} else {
					st.fail("invalid character in substitution index")
				}

				if len(st.str) == 0 {
					st.fail("missing end to substitution index")
				}
				c = st.str[0]
				st.advance(1)
				dec++
			}
			id++
		}

		if id >= len(st.subs) {
			st.failEarlier(fmt.Sprintf("substitution index out of range (%d >= %d)", id, len(st.subs)), dec)
		}

		ret := st.subs[id]

		// We need to update any references to template
		// parameters to refer to the currently active
		// template.
		copy := func(a AST) AST {
			tp, ok := a.(*TemplateParam)
			if !ok {
				return nil
			}
			if len(st.templates) == 0 {
				st.failEarlier("substituted template parameter not in scope of template", dec)
			}
			template := st.templates[len(st.templates)-1]
			if template == nil {
				// This template parameter is within
				// the scope of a cast operator.
				return &TemplateParam{Index: tp.Index, Template: nil}
			}

			if tp.Index >= len(template.Args) {
				st.failEarlier(fmt.Sprintf("substituted template index out of range (%d >= %d)", tp.Index, len(template.Args)), dec)
			}

			return &TemplateParam{Index: tp.Index, Template: template}
		}
		var seen []AST
		skip := func(a AST) bool {
			if _, ok := a.(*Typed); ok {
				return true
			}
			for _, v := range seen {
				if v == a {
					return true
				}
			}
			seen = append(seen, a)
			return false
		}
		if c := ret.Copy(copy, skip); c != nil {
			return c
		}

		return ret
	} else {
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
	var seen []AST
	skip := func(a AST) bool {
		for _, v := range seen {
			if v == a {
				return true
			}
		}
		seen = append(seen, a)
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

				var seen []AST
				skip := func(sub AST) bool {
					// Don't traverse into another
					// pack expansion.
					if _, ok := sub.(*PackExpansion); ok {
						return true
					}
					for _, v := range seen {
						if v == sub {
							return true
						}
					}
					seen = append(seen, sub)
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
	var seen []AST
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
		for _, v := range seen {
			if v == a {
				return false
			}
		}
		seen = append(seen, a)
		return true
	})
	return ret
}
