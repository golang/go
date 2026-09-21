// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"fmt"
	"net/http"
	"reflect"
	_ "unsafe" // for linkname

	. "golang.org/x/net/internal/http3"
)

//go:linkname registerServer net/http_test.registerHTTP3Server
func registerServer(s *http.Server, opts any) error {
	var o ServerOpts
	if err := shallowStructCopy(&o, opts); err != nil {
		return err
	}
	return RegisterServer(s, o)
}

//go:linkname registerTransport net/http_test.registerHTTP3Transport
func registerTransport(tr *http.Transport, opts any) error {
	var o TransportOpts
	if err := shallowStructCopy(&o, opts); err != nil {
		return err
	}
	return RegisterTransport(tr, o)
}

// shallowStructCopy copies every field in src to *dst.
// src must be a struct, and dst must be a pointer to a struct.
//
// We use this to let net/http tests pass their own version of an options struct to
// registerHTTP3{Server,Transport}.
//
// This is a temporary measure pending this package having a public API,
// at which time net/http tests can use the ServerOpts and TransportOpts directly.
func shallowStructCopy(dst, src any) error {
	dv := reflect.ValueOf(dst).Elem()
	sv := reflect.ValueOf(src)
	for i := range sv.Type().NumField() {
		stype := sv.Type().Field(i)
		if !stype.IsExported() {
			return fmt.Errorf("%T contains unexported fields", src)
		}
		sf := sv.Field(i)
		df := dv.FieldByName(stype.Name)
		if !df.CanSet() {
			return fmt.Errorf("%T.%v: field does not exist or is unassignable", dst, stype.Name)
		}
		if !sf.Type().AssignableTo(df.Type()) {
			return fmt.Errorf("%T.%v: %v is not assignable to %v", dst, stype.Name, sf.Type(), df.Type())
		}
		df.Set(sf)
	}
	return nil
}
