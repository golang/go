// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
)

// Instantiate instantiates the type typ with the given type arguments.
// typ must be a *Named or a *Signature type, it must be generic, and
// its number of type parameters must match the number of provided type
// arguments. The result is a new, instantiated (not generic) type of
// the same kind (either a *Named or a *Signature). The type arguments
// are not checked against the constraints of the type parameters.
// Any methods attached to a *Named are simply copied; they are not
// instantiated.
func Instantiate(pos syntax.Pos, typ Type, targs []Type) (res Type) {
	// TODO(gri) This code is basically identical to the prolog
	//           in Checker.instantiate. Factor.
	var tparams []*TypeName
	switch t := typ.(type) {
	case *Named:
		tparams = t.tparams
	case *Signature:
		tparams = t.tparams
		defer func() {
			// If we had an unexpected failure somewhere don't panic below when
			// asserting res.(*Signature). Check for *Signature in case Typ[Invalid]
			// is returned.
			if _, ok := res.(*Signature); !ok {
				return
			}
			// If the signature doesn't use its type parameters, subst
			// will not make a copy. In that case, make a copy now (so
			// we can set tparams to nil w/o causing side-effects).
			if t == res {
				copy := *t
				res = &copy
			}
			// After instantiating a generic signature, it is not generic
			// anymore; we need to set tparams to nil.
			res.(*Signature).tparams = nil
		}()

	default:
		panic(fmt.Sprintf("%v: cannot instantiate %v", pos, typ))
	}

	// the number of supplied types must match the number of type parameters
	if len(targs) != len(tparams) {
		panic(fmt.Sprintf("%v: got %d arguments but %d type parameters", pos, len(targs), len(tparams)))
	}

	if len(tparams) == 0 {
		return typ // nothing to do (minor optimization)
	}

	smap := makeSubstMap(tparams, targs)
	return (*Checker)(nil).subst(pos, typ, smap)
}
