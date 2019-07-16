package complit

var (
	// Multi-line declarations
	V1	= T{
		F1: "hello",
		// contains filtered or unexported fields
	}
	V2	= T{

		F1: "hello",
		// contains filtered or unexported fields
	}
	V3	= T{
		F1:	"hello",
		F2: T2{
			A: "world",
			// contains filtered or unexported fields
		},
		// contains filtered or unexported fields
	}
	V4	= T{
		// contains filtered or unexported fields
	}

	// Single-line declarations
	V5	= T{F1: "hello", /* contains filtered or unexported fields */}
	V6	= T{F1: "hello", /* contains filtered or unexported fields */}
	V7	= T{/* contains filtered or unexported fields */}

	// Mixed-mode declarations
	V8	= T{
		F1:	"hello",
		F3:	"world",
		// contains filtered or unexported fields
	}
	V9	= T{
		F1: "hello",
		// contains filtered or unexported fields
	}
	V10	= T{
		F1:	"hello",

		F4:	"world",
		// contains filtered or unexported fields
	}

	// Other miscellaneous declarations
	V11	= T{
		t{
			A: "world",
			// contains filtered or unexported fields
		},
		// contains filtered or unexported fields
	}
	V12	= T{
		F1:	make(chan int),

		F3:	make(map[int]string),
		// contains filtered or unexported fields
	}
)
