// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"fmt"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestPointer(t *testing.T) {
	tests := []struct {
		in         Pointer
		wantParent Pointer
		wantLast   string
		wantTokens []string
		wantValid  bool
	}{
		{"", "", "", nil, true},
		{"a", "", "a", []string{"a"}, false},
		{"~", "", "~", []string{"~"}, false},
		{"/a", "", "a", []string{"a"}, true},
		{"/foo/bar", "/foo", "bar", []string{"foo", "bar"}, true},
		{"///", "//", "", []string{"", "", ""}, true},
		{"/~0~1", "", "~/", []string{"~/"}, true},
		{"/\xde\xad\xbe\xef", "", "\xde\xad\xbe\xef", []string{"\xde\xad\xbe\xef"}, false},
	}
	for _, tt := range tests {
		if got := tt.in.Parent(); got != tt.wantParent {
			t.Errorf("Pointer(%q).Parent = %q, want %q", tt.in, got, tt.wantParent)
		}
		if got := tt.in.LastToken(); got != tt.wantLast {
			t.Errorf("Pointer(%q).Last = %q, want %q", tt.in, got, tt.wantLast)
		}
		if strings.HasPrefix(string(tt.in), "/") {
			wantRoundtrip := tt.in
			if !utf8.ValidString(string(wantRoundtrip)) {
				// Replace bytes of invalid UTF-8 with Unicode replacement character.
				wantRoundtrip = Pointer([]rune(wantRoundtrip))
			}
			if got := tt.in.Parent().AppendToken(tt.in.LastToken()); got != wantRoundtrip {
				t.Errorf("Pointer(%q).Parent().AppendToken(LastToken()) = %q, want %q", tt.in, got, tt.in)
			}
			in := tt.in
			for {
				if (in + "x").Contains(tt.in) {
					t.Errorf("Pointer(%q).Contains(%q) = true, want false", in+"x", tt.in)
				}
				if !in.Contains(tt.in) {
					t.Errorf("Pointer(%q).Contains(%q) = false, want true", in, tt.in)
				}
				if in == in.Parent() {
					break
				}
				in = in.Parent()
			}
		}
		if got := slices.Collect(tt.in.Tokens()); !slices.Equal(got, tt.wantTokens) {
			t.Errorf("Pointer(%q).Tokens = %q, want %q", tt.in, got, tt.wantTokens)
		}
		if got := tt.in.IsValid(); got != tt.wantValid {
			t.Errorf("Pointer(%q).IsValid = %v, want %v", tt.in, got, tt.wantValid)
		}
	}
}

func TestStateMachine(t *testing.T) {
	// To test a state machine, we pass an ordered sequence of operations and
	// check whether the current state is as expected.
	// The operation type is a union type of various possible operations,
	// which either call mutating methods on the state machine or
	// call accessor methods on state machine and verify the results.
	type operation any
	type (
		// stackLengths checks the results of stateEntry.length accessors.
		stackLengths []int64

		// appendTokens is sequence of token kinds to append where
		// none of them are expected to fail.
		//
		// For example: `[nft]` is equivalent to the following sequence:
		//
		//	pushArray()
		//	appendLiteral()
		//	appendString()
		//	appendNumber()
		//	popArray()
		//
		appendTokens string

		// appendToken is a single token kind to append with the expected error.
		appendToken struct {
			kind Kind
			want error
		}

		// needDelim checks the result of the needDelim accessor.
		needDelim struct {
			next Kind
			want byte
		}
	)

	// Each entry is a sequence of tokens to pass to the state machine.
	tests := []struct {
		label string
		ops   []operation
	}{{
		"TopLevelValues",
		[]operation{
			stackLengths{0},
			needDelim{'n', 0},
			appendTokens(`nft`),
			stackLengths{3},
			needDelim{'"', 0},
			appendTokens(`"0[]{}`),
			stackLengths{7},
		},
	}, {
		"ArrayValues",
		[]operation{
			stackLengths{0},
			needDelim{'[', 0},
			appendTokens(`[`),
			stackLengths{1, 0},
			needDelim{'n', 0},
			appendTokens(`nft`),
			stackLengths{1, 3},
			needDelim{'"', ','},
			appendTokens(`"0[]{}`),
			stackLengths{1, 7},
			needDelim{']', 0},
			appendTokens(`]`),
			stackLengths{1},
		},
	}, {
		"ObjectValues",
		[]operation{
			stackLengths{0},
			needDelim{'{', 0},
			appendTokens(`{`),
			stackLengths{1, 0},
			needDelim{'"', 0},
			appendTokens(`"`),
			stackLengths{1, 1},
			needDelim{'n', ':'},
			appendTokens(`n`),
			stackLengths{1, 2},
			needDelim{'"', ','},
			appendTokens(`"f"t`),
			stackLengths{1, 6},
			appendTokens(`"""0"[]"{}`),
			stackLengths{1, 14},
			needDelim{'}', 0},
			appendTokens(`}`),
			stackLengths{1},
		},
	}, {
		"ObjectCardinality",
		[]operation{
			appendTokens(`{`),

			// Appending any kind other than string for object name is an error.
			appendToken{'n', ErrNonStringName},
			appendToken{'f', ErrNonStringName},
			appendToken{'t', ErrNonStringName},
			appendToken{'0', ErrNonStringName},
			appendToken{'{', ErrNonStringName},
			appendToken{'[', ErrNonStringName},
			appendTokens(`"`),

			// Appending '}' without first appending any value is an error.
			appendToken{'}', errMissingValue},
			appendTokens(`"`),

			appendTokens(`}`),
		},
	}, {
		"MismatchingDelims",
		[]operation{
			appendToken{'}', errMismatchDelim}, // appending '}' without preceding '{'
			appendTokens(`[[{`),
			appendToken{']', errMismatchDelim}, // appending ']' that mismatches preceding '{'
			appendTokens(`}]`),
			appendToken{'}', errMismatchDelim}, // appending '}' that mismatches preceding '['
			appendTokens(`]`),
			appendToken{']', errMismatchDelim}, // appending ']' without preceding '['
		},
	}}

	for _, tt := range tests {
		t.Run(tt.label, func(t *testing.T) {
			// Flatten appendTokens to sequence of appendToken entries.
			var ops []operation
			for _, op := range tt.ops {
				if toks, ok := op.(appendTokens); ok {
					for _, k := range []byte(toks) {
						ops = append(ops, appendToken{Kind(k), nil})
					}
					continue
				}
				ops = append(ops, op)
			}

			// Append each token to the state machine and check the output.
			var state stateMachine
			state.reset()
			var sequence []Kind
			for _, op := range ops {
				switch op := op.(type) {
				case stackLengths:
					var got []int64
					for i := range state.Depth() {
						e := state.index(i)
						got = append(got, e.Length())
					}
					want := []int64(op)
					if !slices.Equal(got, want) {
						t.Fatalf("%s: stack lengths mismatch:\ngot  %v\nwant %v", sequence, got, want)
					}
				case appendToken:
					got := state.append(op.kind)
					if !equalError(got, op.want) {
						t.Fatalf("%s: append('%c') = %v, want %v", sequence, op.kind, got, op.want)
					}
					if got == nil {
						sequence = append(sequence, op.kind)
					}
				case needDelim:
					if got := state.needDelim(op.next); got != op.want {
						t.Fatalf("%s: needDelim('%c') = '%c', want '%c'", sequence, op.next, got, op.want)
					}
				default:
					panic(fmt.Sprintf("unknown operation: %T", op))
				}
			}
		})
	}
}

// append is a thin wrapper over the other append, pop, or push methods
// based on the token kind.
func (s *stateMachine) append(k Kind) error {
	switch k {
	case 'n', 'f', 't':
		return s.appendLiteral()
	case '"':
		return s.appendString()
	case '0':
		return s.appendNumber()
	case '{':
		return s.pushObject()
	case '}':
		return s.popObject()
	case '[':
		return s.pushArray()
	case ']':
		return s.popArray()
	default:
		panic(fmt.Sprintf("invalid token kind: '%c'", k))
	}
}

func TestObjectNamespace(t *testing.T) {
	type operation any
	type (
		insert struct {
			name         string
			wantInserted bool
		}
		removeLast struct{}
	)

	// Sequence of insert operations to perform (order matters).
	ops := []operation{
		insert{`""`, true},
		removeLast{},
		insert{`""`, true},
		insert{`""`, false},

		// Test insertion of the same name with different formatting.
		insert{`"alpha"`, true},
		insert{`"ALPHA"`, true}, // case-sensitive matching
		insert{`"alpha"`, false},
		insert{`"\u0061\u006c\u0070\u0068\u0061"`, false}, // unescapes to "alpha"
		removeLast{},                                      // removes "ALPHA"
		insert{`"alpha"`, false},
		removeLast{}, // removes "alpha"
		insert{`"alpha"`, true},
		removeLast{},

		// Bulk insert simple names.
		insert{`"alpha"`, true},
		insert{`"bravo"`, true},
		insert{`"charlie"`, true},
		insert{`"delta"`, true},
		insert{`"echo"`, true},
		insert{`"foxtrot"`, true},
		insert{`"golf"`, true},
		insert{`"hotel"`, true},
		insert{`"india"`, true},
		insert{`"juliet"`, true},
		insert{`"kilo"`, true},
		insert{`"lima"`, true},
		insert{`"mike"`, true},
		insert{`"november"`, true},
		insert{`"oscar"`, true},
		insert{`"papa"`, true},
		insert{`"quebec"`, true},
		insert{`"romeo"`, true},
		insert{`"sierra"`, true},
		insert{`"tango"`, true},
		insert{`"uniform"`, true},
		insert{`"victor"`, true},
		insert{`"whiskey"`, true},
		insert{`"xray"`, true},
		insert{`"yankee"`, true},
		insert{`"zulu"`, true},

		// Test insertion of invalid UTF-8.
		insert{`"` + "\ufffd" + `"`, true},
		insert{`"` + "\ufffd" + `"`, false},
		insert{`"\ufffd"`, false},         // unescapes to Unicode replacement character
		insert{`"\uFFFD"`, false},         // unescapes to Unicode replacement character
		insert{`"` + "\xff" + `"`, false}, // mangles as Unicode replacement character
		removeLast{},
		insert{`"` + "\ufffd" + `"`, true},

		// Test insertion of unicode characters.
		insert{`"☺☻☹"`, true},
		insert{`"☺☻☹"`, false},
		removeLast{},
		insert{`"☺☻☹"`, true},
	}

	// Execute the sequence of operations twice:
	// 1) on a fresh namespace and 2) on a namespace that has been reset.
	var ns objectNamespace
	wantNames := []string{}
	for _, reset := range []bool{false, true} {
		if reset {
			ns.reset()
			wantNames = nil
		}

		// Execute the operations and ensure the state is consistent.
		for i, op := range ops {
			switch op := op.(type) {
			case insert:
				gotInserted := ns.insertQuoted([]byte(op.name), false)
				if gotInserted != op.wantInserted {
					t.Fatalf("%d: objectNamespace{%v}.insert(%v) = %v, want %v", i, strings.Join(wantNames, " "), op.name, gotInserted, op.wantInserted)
				}
				if gotInserted {
					b, _ := AppendUnquote(nil, []byte(op.name))
					wantNames = append(wantNames, string(b))
				}
			case removeLast:
				ns.removeLast()
				wantNames = wantNames[:len(wantNames)-1]
			default:
				panic(fmt.Sprintf("unknown operation: %T", op))
			}

			// Check that the namespace is consistent.
			gotNames := []string{}
			for i := range ns.length() {
				gotNames = append(gotNames, string(ns.getUnquoted(i)))
			}
			if !slices.Equal(gotNames, wantNames) {
				t.Fatalf("%d: objectNamespace = {%v}, want {%v}", i, strings.Join(gotNames, " "), strings.Join(wantNames, " "))
			}
		}

		// Verify that we have not switched to using a Go map.
		if ns.mapNames != nil {
			t.Errorf("objectNamespace.mapNames = non-nil, want nil")
		}

		// Insert a large number of names.
		for i := range 64 {
			ns.InsertUnquoted([]byte(fmt.Sprintf(`name%d`, i)))
		}

		// Verify that we did switch to using a Go map.
		if ns.mapNames == nil {
			t.Errorf("objectNamespace.mapNames = nil, want non-nil")
		}
	}
}
