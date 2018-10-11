// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packagestest

import (
	"fmt"
	"go/token"
	"reflect"
	"regexp"
	"strings"

	"golang.org/x/tools/go/expect"
)

// Expect invokes the supplied methods for all expectation comments found in
// the exported source files.
//
// All exported go source files are parsed to collect the expectation
// expressions.
// See the documentation for expect.Parse for how the expectations are collected
// and parsed.
//
// The methods are supplied as a map of name to function, and those functions
// will be matched against the expectations by name.
// Markers with no matching function will be skipped, and functions with no
// matching markers will not be invoked.
// As a special case expectations for the mark function will be processed and
// the names can then be used to identify positions in files for all other
// methods invoked.
//
// Method invocation
//
// When invoking a method the expressions in the parameter list need to be
// converted to values to be passed to the method.
// There are a very limited set of types the arguments are allowed to be.
//   expect.Comment : passed the Comment instance being evaluated.
//   string : can be supplied either a string literal or an identifier.
//   int : can only be supplied an integer literal.
//   token.Pos : has a file position calculated as described below.
//   token.Position : has a file position calculated as described below.
//
// Position calculation
//
// There is some extra handling when a parameter is being coerced into a
// token.Pos or token.Position type argument.
//
// If the parameter is an identifier, it will be treated as the name of an
// marker to look up (as if markers were global variables). These markers
// are the results of all "mark" expectations, where the first parameter is
// the name of the marker and the second is the position of the marker.
//
// If it is a string or regular expression, then it will be passed to
// expect.MatchBefore to look up a match in the line at which it was declared.
//
// It is safe to call this repeatedly with different method sets, but it is
// not safe to call it concurrently.
func (e *Exported) Expect(methods map[string]interface{}) error {
	if e.notes == nil {
		notes := []*expect.Note{}
		for _, module := range e.written {
			for _, filename := range module {
				if !strings.HasSuffix(filename, ".go") {
					continue
				}
				l, err := expect.Parse(e.fset, filename, nil)
				if err != nil {
					return fmt.Errorf("Failed to extract expectations: %v", err)
				}
				notes = append(notes, l...)
			}
		}
		e.notes = notes
	}
	if e.markers == nil {
		if err := e.getMarkers(); err != nil {
			return err
		}
	}
	var err error
	ms := make(map[string]method, len(methods))
	for name, f := range methods {
		mi := method{f: reflect.ValueOf(f)}
		mi.converters = make([]converter, mi.f.Type().NumIn())
		for i := 0; i < len(mi.converters); i++ {
			mi.converters[i], err = e.buildConverter(mi.f.Type().In(i))
			if err != nil {
				return fmt.Errorf("invalid method %v: %v", name, err)
			}
		}
		ms[name] = mi
	}
	for _, n := range e.notes {
		mi, ok := ms[n.Name]
		if !ok {
			continue
		}
		params := make([]reflect.Value, len(mi.converters))
		args := n.Args
		for i, convert := range mi.converters {
			params[i], args, err = convert(n, args)
			if err != nil {
				return fmt.Errorf("%v: %v", e.fset.Position(n.Pos), err)
			}
		}
		if len(args) > 0 {
			return fmt.Errorf("%v: unwanted args got %+v extra", e.fset.Position(n.Pos), args)
		}
		//TODO: catch the error returned from the method
		mi.f.Call(params)
	}
	return nil
}

type marker struct {
	name  string
	start token.Pos
	end   token.Pos
}

func (e *Exported) getMarkers() error {
	e.markers = make(map[string]marker)
	for _, n := range e.notes {
		var name string
		var pattern interface{}
		switch {
		case n.Args == nil:
			// simple identifier form
			name = n.Name
			pattern = n.Name
		case n.Name == "mark":
			if len(n.Args) != 2 {
				return fmt.Errorf("%v: expected 2 args to mark, got %v", e.fset.Position(n.Pos), len(n.Args))
			}
			ident, ok := n.Args[0].(expect.Identifier)
			if !ok {
				return fmt.Errorf("%v: expected identifier, got %T", e.fset.Position(n.Pos), n.Args[0])
			}
			name = string(ident)
			pattern = n.Args[1]
		default:
			// not a marker note, so skip it
			continue
		}
		if old, found := e.markers[name]; found {
			return fmt.Errorf("%v: marker %v already exists at %v", e.fset.Position(n.Pos), name, e.fset.Position(old.start))
		}
		start, end, err := expect.MatchBefore(e.fset, e.fileContents, n.Pos, pattern)
		if err != nil {
			return err
		}
		if start == token.NoPos {
			return fmt.Errorf("%v: pattern %s did not match", e.fset.Position(n.Pos), pattern)
		}
		e.markers[name] = marker{
			name:  name,
			start: start,
			end:   end,
		}
	}
	return nil
}

var (
	noteType       = reflect.TypeOf((*expect.Note)(nil))
	identifierType = reflect.TypeOf(expect.Identifier(""))
	posType        = reflect.TypeOf(token.Pos(0))
	positionType   = reflect.TypeOf(token.Position{})
)

// converter converts from a marker's argument parsed from the comment to
// reflect values passed to the method during Invoke.
// It takes the args remaining, and returns the args it did not consume.
// This allows a converter to consume 0 args for well known types, or multiple
// args for compound types.
type converter func(*expect.Note, []interface{}) (reflect.Value, []interface{}, error)

// method is used to track information about Invoke methods that is expensive to
// calculate so that we can work it out once rather than per marker.
type method struct {
	f          reflect.Value // the reflect value of the passed in method
	converters []converter   // the parameter converters for the method
}

// buildConverter works out what function should be used to go from an ast expressions to a reflect
// value of the type expected by a method.
// It is called when only the target type is know, it returns converters that are flexible across
// all supported expression types for that target type.
func (e *Exported) buildConverter(pt reflect.Type) (converter, error) {
	switch {
	case pt == noteType:
		return func(n *expect.Note, args []interface{}) (reflect.Value, []interface{}, error) {
			return reflect.ValueOf(n), args, nil
		}, nil
	case pt == posType:
		return func(n *expect.Note, args []interface{}) (reflect.Value, []interface{}, error) {
			pos, remains, err := e.posConverter(n, args)
			if err != nil {
				return reflect.Value{}, nil, err
			}
			return reflect.ValueOf(pos), remains, nil
		}, nil
	case pt == positionType:
		return func(n *expect.Note, args []interface{}) (reflect.Value, []interface{}, error) {
			pos, remains, err := e.posConverter(n, args)
			if err != nil {
				return reflect.Value{}, nil, err
			}
			return reflect.ValueOf(e.fset.Position(pos)), remains, nil
		}, nil
	case pt == identifierType:
		return func(n *expect.Note, args []interface{}) (reflect.Value, []interface{}, error) {
			arg := args[0]
			args = args[1:]
			switch arg := arg.(type) {
			case expect.Identifier:
				return reflect.ValueOf(arg), args, nil
			default:
				return reflect.Value{}, nil, fmt.Errorf("cannot convert %v to string", arg)
			}
		}, nil
	case pt.Kind() == reflect.String:
		return func(n *expect.Note, args []interface{}) (reflect.Value, []interface{}, error) {
			arg := args[0]
			args = args[1:]
			switch arg := arg.(type) {
			case expect.Identifier:
				return reflect.ValueOf(string(arg)), args, nil
			case string:
				return reflect.ValueOf(arg), args, nil
			default:
				return reflect.Value{}, nil, fmt.Errorf("cannot convert %v to string", arg)
			}
		}, nil
	case pt.Kind() == reflect.Int64:
		return func(n *expect.Note, args []interface{}) (reflect.Value, []interface{}, error) {
			arg := args[0]
			args = args[1:]
			switch arg := arg.(type) {
			case int64:
				return reflect.ValueOf(arg), args, nil
			default:
				return reflect.Value{}, nil, fmt.Errorf("cannot convert %v to int", arg)
			}
		}, nil
	case pt.Kind() == reflect.Bool:
		return func(n *expect.Note, args []interface{}) (reflect.Value, []interface{}, error) {
			arg := args[0]
			args = args[1:]
			b, ok := arg.(bool)
			if !ok {
				return reflect.Value{}, nil, fmt.Errorf("cannot convert %v to bool", arg)
			}
			return reflect.ValueOf(b), args, nil
		}, nil
	default:
		return nil, fmt.Errorf("param has invalid type %v", pt)
	}
}

func (e *Exported) posConverter(n *expect.Note, args []interface{}) (token.Pos, []interface{}, error) {
	if len(args) < 1 {
		return 0, nil, fmt.Errorf("missing argument")
	}
	arg := args[0]
	args = args[1:]
	switch arg := arg.(type) {
	case expect.Identifier:
		// look up an marker by name
		p, ok := e.markers[string(arg)]
		if !ok {
			return 0, nil, fmt.Errorf("cannot find marker %v", arg)
		}
		return p.start, args, nil
	case string:
		p, _, err := expect.MatchBefore(e.fset, e.fileContents, n.Pos, arg)
		if err != nil {
			return 0, nil, err
		}
		if p == token.NoPos {
			return 0, nil, fmt.Errorf("%v: pattern %s did not match", e.fset.Position(n.Pos), arg)
		}
		return p, args, nil
	case *regexp.Regexp:
		p, _, err := expect.MatchBefore(e.fset, e.fileContents, n.Pos, arg)
		if err != nil {
			return 0, nil, err
		}
		if p == token.NoPos {
			return 0, nil, fmt.Errorf("%v: pattern %s did not match", e.fset.Position(n.Pos), arg)
		}
		return p, args, nil
	default:
		return 0, nil, fmt.Errorf("cannot convert %v to pos", arg)
	}
}
