package stdversion

import "reflect"

var _ = reflect.TypeFor[int]() // ERROR "reflect.TypeFor requires go1.22 or later \(module is go1.21\)"
