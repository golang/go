// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"strings"
	"testing"
)

func TestDiscriminator(t *testing.T) {
	x, y, z := newIdent("x"), newIdent("y"), newIdent("z")
	a := NewValue(NewStringExact("a"))
	b := NewValue(NewStringExact("b"))
	c := NewValue(NewStringExact("c"))

	checkDisc := func(t *testing.T, e envSet, want *ident) {
		t.Helper()
		if e.root.kind != envSum {
			t.Fatalf("want sum, got kind %d", e.root.kind)
		}
		if e.root.discriminator != want {
			name := func(id *ident) string {
				if id == nil {
					return "<nil>"
				}
				return id.name
			}
			t.Errorf("want discriminator %s, got %s", name(want), name(e.root.discriminator))
		}
	}

	t.Run("bind", func(t *testing.T) {
		checkDisc(t, topEnv.bind(x, a, b), x)
	})

	// Each environment binds both x and y, so either can be the
	// discriminator.
	xa := topEnv.bind(x, a).bind(y, a)
	xb := topEnv.bind(x, b).bind(y, b)
	xc := topEnv.bind(x, c).bind(y, a)

	t.Run("union", func(t *testing.T) {
		checkDisc(t, unionEnvs(x, xa, xb), x)
		checkDisc(t, unionEnvs(y, xa, xb), y)
	})

	t.Run("union fallback", func(t *testing.T) {
		// The second environment doesn't bind x, but both bind y.
		checkDisc(t, unionEnvs(x, xa, topEnv.bind(y, b)), y)
	})

	t.Run("infer later candidate", func(t *testing.T) {
		// The first operand suggests y, but only the second operand's
		// discriminator, x, is bound by every term.
		u := unionEnvs(y, xa, xb)
		checkDisc(t, envSet{newEnvExprSum(u.root, topEnv.bind(x, b, c).root)}, x)
	})

	t.Run("union no discriminator", func(t *testing.T) {
		// No ident is bound by both environments.
		checkDisc(t, unionEnvs(x, xa, topEnv.bind(z, b)), nil)
	})

	t.Run("flatten", func(t *testing.T) {
		// A sum of sums with the same discriminator has that discriminator.
		u := unionEnvs(x, xa, xb)
		checkDisc(t, envSet{newEnvExprSum(u.root, xc.root)}, x)
		checkDisc(t, envSet{newEnvExprSum(topEnv.bind(x, a).root, topEnv.bind(x, b, c).root)}, x)
	})

	t.Run("substitute", func(t *testing.T) {
		// Substituting another ident keeps the discriminator.
		u := unionEnvs(x, xa, xb, xc)
		checkDisc(t, envSet{u.root.substitute(y, a)}, x)
		// Substituting the discriminator removes it, but the result can
		// have another discriminator.
		xa2 := topEnv.bind(x, a).bind(y, b)
		u = unionEnvs(x, xa, xb, xa2)
		checkDisc(t, envSet{u.root.substitute(x, a)}, y)
	})

	t.Run("unify", func(t *testing.T) {
		read := func(s string) Closure {
			c, err := Read(strings.NewReader(s), "<test>", ReadOpts{})
			if err != nil {
				t.Fatal(err)
			}
			return c
		}
		u, err := Unify(read("!sum [{x: a, y: 1}, {x: b, y: 2}, {x: c, y: 1}]"), read("{y: 1}"))
		if err != nil {
			t.Fatal(err)
		}
		checkDisc(t, u.env, u.val.Domain.(Var).id)
	})
}
