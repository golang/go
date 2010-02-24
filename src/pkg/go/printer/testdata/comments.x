// This is a package for testing comment placement by go/printer.
//
package main


// The SZ struct; it is empty.
type SZ struct{}

// The S0 struct; no field is exported.
type S0 struct {
	// contains unexported fields
}

// The S1 struct; some fields are not exported.
type S1 struct {
	S0
	A, B, C	float	// 3 exported fields
	D	int	// 2 unexported fields
	// contains unexported fields
}

// The S2 struct; all fields are exported.
type S2 struct {
	S1
	A, B, C	float	// 3 exported fields
}

// The IZ interface; it is empty.
type SZ interface{}

// The I0 interface; no method is exported.
type I0 interface {
	// contains unexported methods
}

// The I1 interface; some methods are not exported.
type I1 interface {
	I0
	F(x float) float	// exported methods
	// contains unexported methods
}

// The I2 interface; all methods are exported.
type I2 interface {
	I0
	F(x float) float	// exported method
	G(x float) float	// exported method
}

// The S3 struct; all comments except for the last one must appear in the export.
type S3 struct {
	// lead comment for F1
	F1	int	// line comment for F1
	// lead comment for F2
	F2	int	// line comment for F2
	// contains unexported fields
}
