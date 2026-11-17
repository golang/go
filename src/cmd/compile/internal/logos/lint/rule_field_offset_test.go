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

// Subtracting two immediate field addresses of one struct cancels its base.
// The controls retain reports when the operands do not establish that proof.
func TestArenaFieldOffset(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "protobuf_field_delta",
			body: `o := (uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))); return o`,
		},
		{
			name: "protobuf_delta_plus_index_offset",
			body: `num := uint32(64); o := (uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))) + uintptr(num/32)*unsafe.Sizeof(uint32(0)); return o`,
		},
		{
			name: "returned_delta",
			body: `return uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))`,
		},
		{
			name: "reversed_fields",
			body: `return uintptr(unsafe.Pointer(&frame.d)) - uintptr(unsafe.Pointer(&frame.a))`,
		},
		{
			name: "same_field",
			body: `return uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.a))`,
		},
		{
			name: "parentheses",
			body: `return (uintptr(unsafe.Pointer(&(frame.a)))) - (uintptr(unsafe.Pointer(&(frame.d))))`,
		},
		{
			name: "named_conversion_types",
			body: `o := address(rawPointer(&frame.a)) - address(rawPointer(&frame.d)); return uintptr(o)`,
		},
		{
			name: "stored_delta",
			body: `escaped = uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d)); return escaped`,
		},
		{
			name: "composite_delta",
			body: `v := []uintptr{uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))}; return v[0]`,
		},
		{
			name: "different_objects",
			body: `return uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&other.d))`,
			want: "&frame.a",
		},
		{
			name: "address_sum",
			body: `return uintptr(unsafe.Pointer(&frame.a)) + uintptr(unsafe.Pointer(&frame.d))`,
			want: "&frame.a",
		},
		{
			name: "address_plus_offset",
			body: `return uintptr(unsafe.Pointer(&frame.a)) + 8`,
			want: "&frame.a",
		},
		{
			name: "scaled_operand",
			body: `return 2*uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))`,
			want: "&frame.a",
		},
		{
			name: "masked_operand",
			body: `return (uintptr(unsafe.Pointer(&frame.a)) &^ 7) - uintptr(unsafe.Pointer(&frame.d))`,
			want: "&frame.a",
		},
		{
			name: "adjusted_operands",
			body: `return (uintptr(unsafe.Pointer(&frame.a)) + 8) - (uintptr(unsafe.Pointer(&frame.d)) + 8)`,
			want: "&frame.a",
		},
		{
			name: "mixed_indirect_field",
			body: `p := &other; return uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&p.d))`,
			want: "&frame.a",
		},
		{
			name: "promoted_fields",
			body: `var embedded struct{ layout }; return uintptr(unsafe.Pointer(&embedded.a)) - uintptr(unsafe.Pointer(&embedded.d))`,
			want: "&embedded.a",
		},
		{
			name: "mixed_call",
			body: `return uintptr(unsafe.Pointer(&frame.a)) - observed(uintptr(unsafe.Pointer(&frame.d)))`,
			want: "&frame.a",
		},
		{
			name: "shadowed_uintptr_call",
			body: `uintptr := func(p unsafe.Pointer) integer { return 0 }; return integer(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))`,
			want: "&frame.a",
		},
		{
			name: "pointer_function_call",
			body: `rawPointer := func(p *[1]uint32) unsafe.Pointer { return unsafe.Pointer(p) }; return uintptr(rawPointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))`,
			want: "&frame.d",
		},
		{
			name: "indexed_address",
			body: `var rows [2]layout; return uintptr(unsafe.Pointer(&rows[index()].a)) - uintptr(unsafe.Pointer(&rows[0].d))`,
			want: "&rows[index()].a",
		},
		{
			name: "delta_plus_frame_address",
			body: `return (uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))) + uintptr(unsafe.Pointer(&other.a))`,
			want: "&other.a",
		},
	}
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "offset_unsafe.go"
		level := Deny
		if kind == KindLgo {
			name = "offset_unsafe.lgo"
			level = Forbid
		}
		for _, tt := range tests {
			t.Run(name+"/"+tt.name, func(t *testing.T) {
				src := fieldOffsetHead + "\nfunc target() uintptr { var frame, other layout; _, _ = frame, other; " + tt.body + " }\n"
				in := check(t, kind, name, src)
				got := runRule(in, policyOf(in), collectDirectives(in), arenaPointerEscape)
				wantCount := 0
				if tt.want != "" {
					wantCount = 1
				}
				if wantCount == 0 && len(got) != 0 {
					t.Logf("arena results: %v", summarize(got))
					t.Fatal("relative field offset reports a retained frame address")
				}
				if len(got) != wantCount {
					t.Fatalf("arena results = %v, want %d", summarize(got), wantCount)
				}
				for _, result := range got {
					if result.Rule != arenaPointerEscape.ID || result.Severity != level || !strings.Contains(result.Message, tt.want) {
						t.Errorf("arena result = %+v, want %s %s at %s", result, arenaPointerEscape.ID, level, tt.want)
					}
				}
			})
		}
	}
}

func TestCompileArenaFieldOffset(t *testing.T) {
	for _, name := range []string{"offset_unsafe.go", "offset_unsafe.lgo"} {
		for _, tc := range []struct {
			name string
			body string
			deny bool
		}{
			{"delta", `return uintptr(unsafe.Pointer(&frame.a)) - uintptr(unsafe.Pointer(&frame.d))`, false},
			{"address", `return uintptr(unsafe.Pointer(&frame.a)) + 8`, true},
		} {
			t.Run(name+"/"+tc.name, func(t *testing.T) {
				src := fieldOffsetHead + "\nfunc target() uintptr { var frame layout; " + tc.body + " }\n"
				res := compileFile(t, name, src, "-logos.lintonly")
				if tc.deny {
					lines := strings.Split(strings.TrimSpace(res.output), "\n")
					if res.exitCode != 2 || len(lines) != 1 || !strings.Contains(lines[0], "[correctness::arena_pointer_escape]") || res.objectOK {
						t.Errorf("frame address: exit %d, output %q, object %v", res.exitCode, res.output, res.objectOK)
					}
				} else if res.exitCode != 0 || res.output != "" || res.objectOK {
					t.Errorf("field delta: exit %d, output %q, object %v", res.exitCode, res.output, res.objectOK)
				}
			})
		}
	}
}
