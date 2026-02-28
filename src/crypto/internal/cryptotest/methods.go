// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"fmt"
	"reflect"
	"slices"
	"testing"
)

// NoExtraMethods checks that the concrete type of *ms has no exported methods
// beyond the methods of the interface type of *ms, and any others specified in
// the allowed list.
//
// These methods are accessible through interface upgrades, so they end up part
// of the API even if undocumented per Hyrum's Law.
//
// ms must be a pointer to a non-nil interface.
func NoExtraMethods(t *testing.T, ms any, allowed ...string) {
	t.Helper()
	extraMethods, err := extraMethods(ms)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range extraMethods {
		if slices.Contains(allowed, m) {
			continue
		}
		t.Errorf("unexpected method %q", m)
	}
}

func extraMethods(ip any) ([]string, error) {
	v := reflect.ValueOf(ip)
	if v.Kind() != reflect.Ptr || v.Elem().Kind() != reflect.Interface || v.Elem().IsNil() {
		return nil, fmt.Errorf("argument must be a pointer to a non-nil interface")
	}

	interfaceType := v.Elem().Type()
	concreteType := v.Elem().Elem().Type()

	interfaceMethods := make(map[string]bool)
	for i := range interfaceType.NumMethod() {
		interfaceMethods[interfaceType.Method(i).Name] = true
	}

	var extraMethods []string
	for i := range concreteType.NumMethod() {
		m := concreteType.Method(i)
		if !m.IsExported() {
			continue
		}
		if !interfaceMethods[m.Name] {
			extraMethods = append(extraMethods, m.Name)
		}
	}

	return extraMethods, nil
}
