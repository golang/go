// Test of examples with divergent packages.

package buf_test

func Example() {} // OK because is package-level.

func Example_suffix() {} // OK because refers to suffix annotation.

func Example_BadSuffix() {} // ERROR "Example_BadSuffix has malformed example suffix: BadSuffix"

func ExampleBuf() {} // OK because refers to known top-level type.

func ExampleBuf_Append() {} // OK because refers to known method.

func ExampleBuf_Clear() {} // ERROR "ExampleBuf_Clear refers to unknown field or method: Buf.Clear"

func ExampleBuf_suffix() {} // OK because refers to suffix annotation.

func ExampleBuf_Append_Bad() {} // ERROR "ExampleBuf_Append_Bad has malformed example suffix: Bad"

func ExampleBuf_Append_suffix() {} // OK because refers to known method with valid suffix.

func ExampleDefaultBuf() {} // OK because refers to top-level identifier.

func ExampleBuf_Reset() bool { return true } // ERROR "ExampleBuf_Reset should return nothing"

func ExampleBuf_Len(i int) {} // ERROR "ExampleBuf_Len should be niladic"

// "Puffer" is German for "Buffer".

func ExamplePuffer() {} // ERROR "ExamplePuffer refers to unknown identifier: Puffer"

func ExamplePuffer_Append() {} // ERROR "ExamplePuffer_Append refers to unknown identifier: Puffer"

func ExamplePuffer_suffix() {} // ERROR "ExamplePuffer_suffix refers to unknown identifier: Puffer"
