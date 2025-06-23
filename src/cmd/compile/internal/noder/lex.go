// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"
	"internal/buildcfg"
	"strings"

	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
)

func isSpace(c rune) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

func isQuoted(s string) bool {
	return len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"'
}

const (
	funcPragmas = ir.Nointerface |
		ir.Noescape |
		ir.Norace |
		ir.Nosplit |
		ir.Noinline |
		ir.NoCheckPtr |
		ir.RegisterParams | // TODO(register args) remove after register abi is working
		ir.CgoUnsafeArgs |
		ir.UintptrKeepAlive |
		ir.UintptrEscapes |
		ir.Systemstack |
		ir.Nowritebarrier |
		ir.Nowritebarrierrec |
		ir.Yeswritebarrierrec
)

func pragmaFlag(verb string) ir.PragmaFlag {
	switch verb {
	case "go:build":
		return ir.GoBuildPragma
	case "go:nointerface":
		if buildcfg.Experiment.FieldTrack {
			return ir.Nointerface
		}
	case "go:noescape":
		return ir.Noescape
	case "go:norace":
		return ir.Norace
	case "go:nosplit":
		return ir.Nosplit | ir.NoCheckPtr // implies NoCheckPtr (see #34972)
	case "go:noinline":
		return ir.Noinline
	case "go:nocheckptr":
		return ir.NoCheckPtr
	case "go:systemstack":
		return ir.Systemstack
	case "go:nowritebarrier":
		return ir.Nowritebarrier
	case "go:nowritebarrierrec":
		return ir.Nowritebarrierrec | ir.Nowritebarrier // implies Nowritebarrier
	case "go:yeswritebarrierrec":
		return ir.Yeswritebarrierrec
	case "go:cgo_unsafe_args":
		return ir.CgoUnsafeArgs | ir.NoCheckPtr // implies NoCheckPtr (see #34968)
	case "go:uintptrkeepalive":
		return ir.UintptrKeepAlive
	case "go:uintptrescapes":
		// This directive extends //go:uintptrkeepalive by forcing
		// uintptr arguments to escape to the heap, which makes stack
		// growth safe.
		return ir.UintptrEscapes | ir.UintptrKeepAlive // implies UintptrKeepAlive
	case "go:registerparams": // TODO(register args) remove after register abi is working
		return ir.RegisterParams
	}
	return 0
}

// pragcgo is called concurrently if files are parsed concurrently.
func (p *noder) pragcgo(pos syntax.Pos, text string) {
	f := pragmaFields(text)

	verb := strings.TrimPrefix(f[0], "go:")
	f[0] = verb

	switch verb {
	case "cgo_export_static", "cgo_export_dynamic":
		switch {
		case len(f) == 2 && !isQuoted(f[1]):
		case len(f) == 3 && !isQuoted(f[1]) && !isQuoted(f[2]):
		default:
			p.error(syntax.Error{Pos: pos, Msg: fmt.Sprintf(`usage: //go:%s local [remote]`, verb)})
			return
		}
	case "cgo_import_dynamic":
		switch {
		case len(f) == 2 && !isQuoted(f[1]):
		case len(f) == 3 && !isQuoted(f[1]) && !isQuoted(f[2]):
		case len(f) == 4 && !isQuoted(f[1]) && !isQuoted(f[2]) && isQuoted(f[3]):
			f[3] = strings.Trim(f[3], `"`)
			if buildcfg.GOOS == "aix" && f[3] != "" {
				// On Aix, library pattern must be "lib.a/object.o"
				// or "lib.a/libname.so.X"
				n := strings.Split(f[3], "/")
				if len(n) != 2 || !strings.HasSuffix(n[0], ".a") || (!strings.HasSuffix(n[1], ".o") && !strings.Contains(n[1], ".so.")) {
					p.error(syntax.Error{Pos: pos, Msg: `usage: //go:cgo_import_dynamic local [remote ["lib.a/object.o"]]`})
					return
				}
			}
		default:
			p.error(syntax.Error{Pos: pos, Msg: `usage: //go:cgo_import_dynamic local [remote ["library"]]`})
			return
		}
	case "cgo_import_static":
		switch {
		case len(f) == 2 && !isQuoted(f[1]):
		default:
			p.error(syntax.Error{Pos: pos, Msg: `usage: //go:cgo_import_static local`})
			return
		}
	case "cgo_dynamic_linker":
		switch {
		case len(f) == 2 && isQuoted(f[1]):
			f[1] = strings.Trim(f[1], `"`)
		default:
			p.error(syntax.Error{Pos: pos, Msg: `usage: //go:cgo_dynamic_linker "path"`})
			return
		}
	case "cgo_ldflag":
		switch {
		case len(f) == 2 && isQuoted(f[1]):
			f[1] = strings.Trim(f[1], `"`)
		default:
			p.error(syntax.Error{Pos: pos, Msg: `usage: //go:cgo_ldflag "arg"`})
			return
		}
	default:
		return
	}
	p.pragcgobuf = append(p.pragcgobuf, f)
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
