// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"strings"
	"testing"
)

const fieldOffsetHead = `package p

import "unsafe"

type layout struct {
	d [8]byte
	a [1]uint32
}

type address uintptr
type rawPointer = unsafe.Pointer
type integer = uintptr

var escaped uintptr

func observed(v uintptr) uintptr { return v }
func index() int { return 0 }
`

func TestArenaFieldOffset(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tests := []struct {
		name string
		body string
	}{
		{"protobuf_field_delta", `o := (uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))); return o`},
		{"protobuf_delta_plus_index_offset", `num := uint32(64); o := (uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))) + uintptr(num/32)*unsafe.Sizeof(uint32(0)); return o`},
		{"returned_delta", `return uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))`},
		{"reversed_fields", `return uintptr(unsafe.Pointer(&frame.d)) - uintptr(unsafe.Pointer(&frame.a))`},
		{"same_field", `return uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.a))`},
		{"parentheses", `return (uintptr(unsafe.Pointer(&(frame.a)))) - (uintptr(unsafe.Pointer(&(frame.d))))`},
		{"named_conversion_types", `o := address(rawPointer(&frame.a)) - address(rawPointer(&frame.d)); return uintptr(o)`},
		{"stored_delta", `escaped = uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d)); return escaped`},
		{"composite_delta", `v := []uintptr{uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))}; return v[0]`},
		{"different_objects", `return uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&other.d))`},
		{"address_sum", `return uintptr(unsafe.Pointer(&frame.a)) + uintptr(unsafe.Pointer(&frame.d))`},
		{"address_plus_offset", `return uintptr(unsafe.Pointer(&frame.a)) + 8`},
		{"scaled_operand", `return 2*uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))`},
		{"masked_operand", `return (uintptr(unsafe.Pointer(&frame.a)) &^ 7) - uintptr(unsafe.Pointer(&frame.d))`},
		{"adjusted_operands", `return (uintptr(unsafe.Pointer(&frame.a)) + 8) - (uintptr(unsafe.Pointer(&frame.d)) + 8)`},
		{"mixed_indirect_field", `p := &other; return uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&p.d))`},
		{"promoted_fields", `var embedded struct{ layout }; return uintptr(unsafe.Pointer(&embedded.a)) - uintptr(unsafe.Pointer(&embedded.d))`},
		{"mixed_call", `return uintptr(unsafe.Pointer(&frame.a)) - observed(uintptr(unsafe.Pointer(&frame.d)))`},
		{"shadowed_uintptr_call", `uintptr := func(p unsafe.Pointer) integer { return 0 }; return integer(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))`},
		{"pointer_function_call", `rawPointer := func(p *[1]uint32) unsafe.Pointer { return unsafe.Pointer(p) }; return uintptr(rawPointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))`},
		{"indexed_address", `var rows [2]layout; return uintptr(unsafe.Pointer(&rows[index()].a)) - uintptr(unsafe.Pointer(&rows[0].d))`},
		{"delta_plus_frame_address", `return (uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))) + uintptr(unsafe.Pointer(&other.a))`},
	}
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "offset_unsafe.go"
		if kind == KindLgo {
			name = "offset_unsafe.lgo"
		}
		for _, tt := range tests {
			t.Run(name+"/"+tt.name, func(t *testing.T) {
				src := fieldOffsetHead + "\nfunc target() uintptr { var frame, other layout; _, _ = frame, other; " + tt.body + " }\n"
				in := check(t, kind, name, src)
				if got := runRule(in, policyOf(in), collectDirectives(in), arenaPointerEscape); len(got) != 0 {
					t.Fatalf("integer-only address arithmetic reported: %+v", got)
				}
			})
		}
	}
}

func TestCompileArenaFieldOffset(t *testing.T) {
	for _, name := range []string{"offset_unsafe.go", "offset_unsafe.lgo"} {
		for _, tc := range []struct {
			name   string
			result string
			body   string
			deny   bool
		}{
			{"delta", "uintptr", `return uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))`, false},
			{"address", "uintptr", `return uintptr(unsafe.Pointer(&frame.a)) + 8`, false},
			{"reconstruction", "unsafe.Pointer", `u := uintptr(unsafe.Pointer(&frame.a)); return unsafe.Pointer(u)`, true},
		} {
			t.Run(name+"/"+tc.name, func(t *testing.T) {
				src := fieldOffsetHead + "\nfunc target() " + tc.result + " { var frame layout; " + tc.body + " }\n"
				res := compileFile(t, name, src, "-logos.lintonly")
				if tc.deny {
					lines := strings.Split(strings.TrimSpace(res.output), "\n")
					if res.exitCode != 2 || len(lines) != 1 || !strings.Contains(lines[0], "[correctness::arena_pointer_escape]") || res.objectOK {
						t.Errorf("pointer reconstruction: exit %d, output %q, object %v", res.exitCode, res.output, res.objectOK)
					}
				} else if res.exitCode != 0 || res.output != "" || res.objectOK {
					t.Errorf("integer address: exit %d, output %q, object %v", res.exitCode, res.output, res.objectOK)
				}
			})
		}
	}
}
