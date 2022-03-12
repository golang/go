// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"go/ast"
	"go/constant"
	"go/types"
	"strconv"
	"strings"
	"unicode/utf8"
)

// printfArgKind returns the expected objKind when completing a
// printf-like operand. call is the printf-like function call, and
// argIdx is the index of call.Args being completed.
func printfArgKind(info *types.Info, call *ast.CallExpr, argIdx int) objKind {
	// Printf-like function name must end in "f".
	fn := exprObj(info, call.Fun)
	if fn == nil || !strings.HasSuffix(fn.Name(), "f") {
		return kindAny
	}

	sig, _ := fn.Type().(*types.Signature)
	if sig == nil {
		return kindAny
	}

	// Must be variadic and take at least two params.
	numParams := sig.Params().Len()
	if !sig.Variadic() || numParams < 2 || argIdx < numParams-1 {
		return kindAny
	}

	// Param preceding variadic args must be a (format) string.
	if !types.Identical(sig.Params().At(numParams-2).Type(), types.Typ[types.String]) {
		return kindAny
	}

	// Format string must be a constant.
	strArg := info.Types[call.Args[numParams-2]].Value
	if strArg == nil || strArg.Kind() != constant.String {
		return kindAny
	}

	return formatOperandKind(constant.StringVal(strArg), argIdx-(numParams-1)+1)
}

// formatOperandKind returns the objKind corresponding to format's
// operandIdx'th operand.
func formatOperandKind(format string, operandIdx int) objKind {
	var (
		prevOperandIdx int
		kind           = kindAny
	)
	for {
		i := strings.Index(format, "%")
		if i == -1 {
			break
		}

		var operands []formatOperand
		format, operands = parsePrintfVerb(format[i+1:], prevOperandIdx)

		// Check if any this verb's operands correspond to our target
		// operandIdx.
		for _, v := range operands {
			if v.idx == operandIdx {
				if kind == kindAny {
					kind = v.kind
				} else if v.kind != kindAny {
					// If multiple verbs refer to the same operand, take the
					// intersection of their kinds.
					kind &= v.kind
				}
			}

			prevOperandIdx = v.idx
		}
	}
	return kind
}

type formatOperand struct {
	// idx is the one-based printf operand index.
	idx int
	// kind is a mask of expected kinds of objects for this operand.
	kind objKind
}

// parsePrintfVerb parses the leading printf verb in f. The opening
// "%" must already be trimmed from f. prevIdx is the previous
// operand's index, or zero if this is the first verb. The format
// string is returned with the leading verb removed. Multiple operands
// can be returned in the case of dynamic widths such as "%*.*f".
func parsePrintfVerb(f string, prevIdx int) (string, []formatOperand) {
	var verbs []formatOperand

	addVerb := func(k objKind) {
		verbs = append(verbs, formatOperand{
			idx:  prevIdx + 1,
			kind: k,
		})
		prevIdx++
	}

	for len(f) > 0 {
		// Trim first rune off of f so we are guaranteed to make progress.
		r, l := utf8.DecodeRuneInString(f)
		f = f[l:]

		// We care about three things:
		// 1. The verb, which maps directly to object kind.
		// 2. Explicit operand indices like "%[2]s".
		// 3. Dynamic widths using "*".
		switch r {
		case '%':
			return f, nil
		case '*':
			addVerb(kindInt)
			continue
		case '[':
			// Parse operand index as in "%[2]s".
			i := strings.Index(f, "]")
			if i == -1 {
				return f, nil
			}

			idx, err := strconv.Atoi(f[:i])
			f = f[i+1:]
			if err != nil {
				return f, nil
			}

			prevIdx = idx - 1
			continue
		case 'v', 'T':
			addVerb(kindAny)
		case 't':
			addVerb(kindBool)
		case 'c', 'd', 'o', 'O', 'U':
			addVerb(kindInt)
		case 'e', 'E', 'f', 'F', 'g', 'G':
			addVerb(kindFloat | kindComplex)
		case 'b':
			addVerb(kindInt | kindFloat | kindComplex | kindBytes)
		case 'q', 's':
			addVerb(kindString | kindBytes | kindStringer | kindError)
		case 'x', 'X':
			// Omit kindStringer and kindError though technically allowed.
			addVerb(kindString | kindBytes | kindInt | kindFloat | kindComplex)
		case 'p':
			addVerb(kindPtr | kindSlice)
		case 'w':
			addVerb(kindError)
		case '+', '-', '#', ' ', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			// Flag or numeric width/precision value.
			continue
		default:
			// Assume unrecognized rune is a custom fmt.Formatter verb.
			addVerb(kindAny)
		}

		if len(verbs) > 0 {
			break
		}
	}

	return f, verbs
}
