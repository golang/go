package a_test

import _ "a"

func ExampleFoo() {} // OK because a.Foo exists

func ExampleBar() {} // want "ExampleBar refers to unknown identifier: Bar"
