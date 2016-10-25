// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/syntax"
	"cmd/internal/obj"
	"fmt"
	"strings"
)

// lexlineno is the line number _after_ the most recently read rune.
// In particular, it's advanced (or rewound) as newlines are read (or unread).
var lexlineno int32

// lineno is the line number at the start of the most recently lexed token.
var lineno int32

func isSpace(c rune) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

func isQuoted(s string) bool {
	return len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"'
}

func plan9quote(s string) string {
	if s == "" {
		return "''"
	}
	for _, c := range s {
		if c <= ' ' || c == '\'' {
			return "'" + strings.Replace(s, "'", "''", -1) + "'"
		}
	}
	return s
}

type Pragma syntax.Pragma

const (
	// Func pragmas.
	Nointerface    Pragma = 1 << iota
	Noescape              // func parameters don't escape
	Norace                // func must not have race detector annotations
	Nosplit               // func should not execute on separate stack
	Noinline              // func should not be inlined
	CgoUnsafeArgs         // treat a pointer to one arg as a pointer to them all
	UintptrEscapes        // pointers converted to uintptr escape

	// Runtime-only func pragmas.
	// See ../../../../runtime/README.md for detailed descriptions.
	Systemstack        // func must run on system stack
	Nowritebarrier     // emit compiler error instead of write barrier
	Nowritebarrierrec  // error on write barrier in this or recursive callees
	Yeswritebarrierrec // cancels Nowritebarrierrec in this function and callees

	// Runtime-only type pragmas
	NotInHeap // values of this type must not be heap allocated
)

func pragmaValue(verb string) Pragma {
	switch verb {
	case "go:nointerface":
		if obj.Fieldtrack_enabled != 0 {
			return Nointerface
		}
	case "go:noescape":
		return Noescape
	case "go:norace":
		return Norace
	case "go:nosplit":
		return Nosplit
	case "go:noinline":
		return Noinline
	case "go:systemstack":
		if !compiling_runtime {
			yyerror("//go:systemstack only allowed in runtime")
		}
		return Systemstack
	case "go:nowritebarrier":
		if !compiling_runtime {
			yyerror("//go:nowritebarrier only allowed in runtime")
		}
		return Nowritebarrier
	case "go:nowritebarrierrec":
		if !compiling_runtime {
			yyerror("//go:nowritebarrierrec only allowed in runtime")
		}
		return Nowritebarrierrec | Nowritebarrier // implies Nowritebarrier
	case "go:yeswritebarrierrec":
		if !compiling_runtime {
			yyerror("//go:yeswritebarrierrec only allowed in runtime")
		}
		return Yeswritebarrierrec
	case "go:cgo_unsafe_args":
		return CgoUnsafeArgs
	case "go:uintptrescapes":
		// For the next function declared in the file
		// any uintptr arguments may be pointer values
		// converted to uintptr. This directive
		// ensures that the referenced allocated
		// object, if any, is retained and not moved
		// until the call completes, even though from
		// the types alone it would appear that the
		// object is no longer needed during the
		// call. The conversion to uintptr must appear
		// in the argument list.
		// Used in syscall/dll_windows.go.
		return UintptrEscapes
	case "go:notinheap":
		return NotInHeap
	}
	return 0
}

var internedStrings = map[string]string{}

func internString(b []byte) string {
	s, ok := internedStrings[string(b)] // string(b) here doesn't allocate
	if !ok {
		s = string(b)
		internedStrings[s] = s
	}
	return s
}

func pragcgo(text string) string {
	f := pragmaFields(text)

	verb := f[0][3:] // skip "go:"
	switch verb {
	case "cgo_export_static", "cgo_export_dynamic":
		switch {
		case len(f) == 2 && !isQuoted(f[1]):
			local := plan9quote(f[1])
			return fmt.Sprintln(verb, local)

		case len(f) == 3 && !isQuoted(f[1]) && !isQuoted(f[2]):
			local := plan9quote(f[1])
			remote := plan9quote(f[2])
			return fmt.Sprintln(verb, local, remote)

		default:
			yyerror(`usage: //go:%s local [remote]`, verb)
		}
	case "cgo_import_dynamic":
		switch {
		case len(f) == 2 && !isQuoted(f[1]):
			local := plan9quote(f[1])
			return fmt.Sprintln(verb, local)

		case len(f) == 3 && !isQuoted(f[1]) && !isQuoted(f[2]):
			local := plan9quote(f[1])
			remote := plan9quote(f[2])
			return fmt.Sprintln(verb, local, remote)

		case len(f) == 4 && !isQuoted(f[1]) && !isQuoted(f[2]) && isQuoted(f[3]):
			local := plan9quote(f[1])
			remote := plan9quote(f[2])
			library := plan9quote(strings.Trim(f[3], `"`))
			return fmt.Sprintln(verb, local, remote, library)

		default:
			yyerror(`usage: //go:cgo_import_dynamic local [remote ["library"]]`)
		}
	case "cgo_import_static":
		switch {
		case len(f) == 2 && !isQuoted(f[1]):
			local := plan9quote(f[1])
			return fmt.Sprintln(verb, local)

		default:
			yyerror(`usage: //go:cgo_import_static local`)
		}
	case "cgo_dynamic_linker":
		switch {
		case len(f) == 2 && isQuoted(f[1]):
			path := plan9quote(strings.Trim(f[1], `"`))
			return fmt.Sprintln(verb, path)

		default:
			yyerror(`usage: //go:cgo_dynamic_linker "path"`)
		}
	case "cgo_ldflag":
		switch {
		case len(f) == 2 && isQuoted(f[1]):
			arg := plan9quote(strings.Trim(f[1], `"`))
			return fmt.Sprintln(verb, arg)

		default:
			yyerror(`usage: //go:cgo_ldflag "arg"`)
		}
	}
	return ""
}

// pragmaFields is similar to strings.FieldsFunc(s, isSpace)
// but does not split when inside double quoted regions and always
// splits before the start and after the end of a double quoted region.
// pragmaFields does not recognize escaped quotes. If a quote in s is not
// closed the part after the opening quote will not be returned as a field.
func pragmaFields(s string) []string {
	var a []string
	inQuote := false
	fieldStart := -1 // Set to -1 when looking for start of field.
	for i, c := range s {
		switch {
		case c == '"':
			if inQuote {
				inQuote = false
				a = append(a, s[fieldStart:i+1])
				fieldStart = -1
			} else {
				inQuote = true
				if fieldStart >= 0 {
					a = append(a, s[fieldStart:i])
				}
				fieldStart = i
			}
		case !inQuote && isSpace(c):
			if fieldStart >= 0 {
				a = append(a, s[fieldStart:i])
				fieldStart = -1
			}
		default:
			if fieldStart == -1 {
				fieldStart = i
			}
		}
	}
	if !inQuote && fieldStart >= 0 { // Last field might end at the end of the string.
		a = append(a, s[fieldStart:])
	}
	return a
}
