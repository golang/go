// Test of examples.

package testdata

func Example() {} // OK because is package-level.

func Example_suffix() // OK because refers to suffix annotation.

func Example_BadSuffix() // OK because non-test package was excluded.  No false positives wanted.

func ExampleBuf() // OK because non-test package was excluded.  No false positives wanted.

func ExampleBuf_Append() {} // OK because non-test package was excluded.  No false positives wanted.

func ExampleBuf_Clear() {} // OK because non-test package was excluded.  No false positives wanted.

func ExampleBuf_suffix() {} // OK because refers to suffix annotation.

func ExampleBuf_Append_Bad() {} // OK because non-test package was excluded.  No false positives wanted.

func ExampleBuf_Append_suffix() {} // OK because refers to known method with valid suffix.

func ExampleBuf_Reset() bool { return true } // ERROR "ExampleBuf_Reset should return nothing"

func ExampleBuf_Len(i int) {} // ERROR "ExampleBuf_Len should be niladic"

// "Puffer" is German for "Buffer".

func ExamplePuffer() // OK because non-test package was excluded.  No false positives wanted.

func ExamplePuffer_Append() // OK because non-test package was excluded.  No false positives wanted.

func ExamplePuffer_suffix() // OK because non-test package was excluded.  No false positives wanted.
