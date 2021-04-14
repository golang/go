// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "./f1"

func main() {
	f := f1.Foo{
		doneChan:      nil, // ERROR "cannot refer to unexported field 'doneChan' in struct literal of type f1.Foo"
		DoneChan:      nil, // ERROR "unknown field 'DoneChan' in struct literal of type f1.Foo"
		Name:          "hey",
		name:          "there",   // ERROR "unknown field 'name' in struct literal of type f1.Foo .but does have Name."
		noSuchPrivate: true,      // ERROR "unknown field 'noSuchPrivate' in struct literal of type f1.Foo"
		NoSuchPublic:  true,      // ERROR "unknown field 'NoSuchPublic' in struct literal of type f1.Foo"
		foo:           true,      // ERROR "unknown field 'foo' in struct literal of type f1.Foo"
		hook:          func() {}, // ERROR "cannot refer to unexported field 'hook' in struct literal of type f1.Foo"
		unexported:    func() {}, // ERROR "unknown field 'unexported' in struct literal of type f1.Foo"
		Exported:      func() {}, // ERROR "unknown field 'Exported' in struct literal of type f1.Foo"
	}
	f.doneChan = nil // ERROR "f.doneChan undefined .cannot refer to unexported field or method doneChan."
	f.DoneChan = nil // ERROR "f.DoneChan undefined .type f1.Foo has no field or method DoneChan."
	f.name = nil     // ERROR "f.name undefined .type f1.Foo has no field or method name, but does have Name."

	_ = f.doneChan // ERROR "f.doneChan undefined .cannot refer to unexported field or method doneChan."
	_ = f.DoneChan // ERROR "f.DoneChan undefined .type f1.Foo has no field or method DoneChan."
	_ = f.Name
	_ = f.name          // ERROR "f.name undefined .type f1.Foo has no field or method name, but does have Name."
	_ = f.noSuchPrivate // ERROR "f.noSuchPrivate undefined .type f1.Foo has no field or method noSuchPrivate."
	_ = f.NoSuchPublic  // ERROR "f.NoSuchPublic undefined .type f1.Foo has no field or method NoSuchPublic."
	_ = f.foo           // ERROR "f.foo undefined .type f1.Foo has no field or method foo."
	_ = f.Exported
	_ = f.exported    // ERROR "f.exported undefined .type f1.Foo has no field or method exported, but does have Exported."
	_ = f.Unexported  // ERROR "f.Unexported undefined .type f1.Foo has no field or method Unexported."
	_ = f.unexported  // ERROR "f.unexported undefined .cannot refer to unexported field or method f1..\*Foo..unexported."
	f.unexported = 10 // ERROR "f.unexported undefined .cannot refer to unexported field or method f1..\*Foo..unexported."
	f.unexported()    // ERROR "f.unexported undefined .cannot refer to unexported field or method f1..\*Foo..unexported."
	_ = f.hook        // ERROR "f.hook undefined .cannot refer to unexported field or method hook."
}
