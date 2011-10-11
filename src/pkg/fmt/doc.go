// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	Package fmt implements formatted I/O with functions analogous
	to C's printf and scanf.  The format 'verbs' are derived from C's but
	are simpler.

	Printing:

	The verbs:

	General:
		%v	the value in a default format.
			when printing structs, the plus flag (%+v) adds field names
		%#v	a Go-syntax representation of the value
		%T	a Go-syntax representation of the type of the value
		%%	a literal percent sign; consumes no value

	Boolean:
		%t	the word true or false
	Integer:
		%b	base 2
		%c	the character represented by the corresponding Unicode code point
		%d	base 10
		%o	base 8
		%q	a single-quoted character literal safely escaped with Go syntax.
		%x	base 16, with lower-case letters for a-f
		%X	base 16, with upper-case letters for A-F
		%U	Unicode format: U+1234; same as "U+%04X"
	Floating-point and complex constituents:
		%b	decimalless scientific notation with exponent a power
			of two, in the manner of strconv.Ftoa32, e.g. -123456p-78
		%e	scientific notation, e.g. -1234.456e+78
		%E	scientific notation, e.g. -1234.456E+78
		%f	decimal point but no exponent, e.g. 123.456
		%g	whichever of %e or %f produces more compact output
		%G	whichever of %E or %f produces more compact output
	String and slice of bytes:
		%s	the uninterpreted bytes of the string or slice
		%q	a double-quoted string safely escaped with Go syntax
		%x	base 16, lower-case, two characters per byte
		%X	base 16, upper-case, two characters per byte
	Pointer:
		%p	base 16 notation, with leading 0x

	There is no 'u' flag.  Integers are printed unsigned if they have unsigned type.
	Similarly, there is no need to specify the size of the operand (int8, int64).

	The width and precision control formatting and are in units of Unicode
	code points.  (This differs from C's printf where the units are numbers
	of bytes.) Either or both of the flags may be replaced with the
	character '*', causing their values to be obtained from the next
	operand, which must be of type int.

	For numeric values, width sets the width of the field and precision
	sets the number of places after the decimal, if appropriate.  For
	example, the format %6.2f prints 123.45.

	For strings, width is the minimum number of characters to output,
	padding with spaces if necessary, and precision is the maximum
	number of characters to output, truncating if necessary.

	Other flags:
		+	always print a sign for numeric values;
			guarantee ASCII-only output for %q (%+q)
		-	pad with spaces on the right rather than the left (left-justify the field)
		#	alternate format: add leading 0 for octal (%#o), 0x for hex (%#x);
			0X for hex (%#X); suppress 0x for %p (%#p);
			print a raw (backquoted) string if possible for %q (%#q);
			write e.g. U+0078 'x' if the character is printable for %U (%#U).
		' '	(space) leave a space for elided sign in numbers (% d);
			put spaces between bytes printing strings or slices in hex (% x, % X)
		0	pad with leading zeros rather than spaces

	For each Printf-like function, there is also a Print function
	that takes no format and is equivalent to saying %v for every
	operand.  Another variant Println inserts blanks between
	operands and appends a newline.

	Regardless of the verb, if an operand is an interface value,
	the internal concrete value is used, not the interface itself.
	Thus:
		var i interface{} = 23
		fmt.Printf("%v\n", i)
	will print 23.

	If an operand implements interface Formatter, that interface
	can be used for fine control of formatting.

	If an operand implements method String() string that method
	will be used to convert the object to a string, which will then
	be formatted as required by the verb (if any). To avoid
	recursion in cases such as
		type X int
		func (x X) String() string { return Sprintf("%d", x) }
	cast the value before recurring:
		func (x X) String() string { return Sprintf("%d", int(x)) }

	Format errors:

	If an invalid argument is given for a verb, such as providing
	a string to %d, the generated string will contain a
	description of the problem, as in these examples:

		Wrong type or unknown verb: %!verb(type=value)
			Printf("%d", hi):          %!d(string=hi)
		Too many arguments: %!(EXTRA type=value)
			Printf("hi", "guys"):      hi%!(EXTRA string=guys)
		Too few arguments: %!verb(MISSING)
			Printf("hi%d"):            hi %!d(MISSING)
		Non-int for width or precision: %!(BADWIDTH) or %!(BADPREC)
			Printf("%*s", 4.5, "hi"):  %!(BADWIDTH)hi
			Printf("%.*s", 4.5, "hi"): %!(BADPREC)hi

	All errors begin with the string "%!" followed sometimes
	by a single character (the verb) and end with a parenthesized
	description.

	Scanning:

	An analogous set of functions scans formatted text to yield
	values.  Scan, Scanf and Scanln read from os.Stdin; Fscan,
	Fscanf and Fscanln read from a specified io.Reader; Sscan,
	Sscanf and Sscanln read from an argument string.  Scanln,
	Fscanln and Sscanln stop scanning at a newline and require that
	the items be followed by one; Sscanf, Fscanf and Sscanf require
	newlines in the input to match newlines in the format; the other
	routines treat newlines as spaces.

	Scanf, Fscanf, and Sscanf parse the arguments according to a
	format string, analogous to that of Printf.  For example, %x
	will scan an integer as a hexadecimal number, and %v will scan
	the default representation format for the value.

	The formats behave analogously to those of Printf with the
	following exceptions:

		%p is not implemented
		%T is not implemented
		%e %E %f %F %g %G are all equivalent and scan any floating point or complex value
		%s and %v on strings scan a space-delimited token

	The familiar base-setting prefixes 0 (octal) and 0x
	(hexadecimal) are accepted when scanning integers without a
	format or with the %v verb.

	Width is interpreted in the input text (%5s means at most
	five runes of input will be read to scan a string) but there
	is no syntax for scanning with a precision (no %5.2f, just
	%5f).

	When scanning with a format, all non-empty runs of space
	characters (except newline) are equivalent to a single
	space in both the format and the input.  With that proviso,
	text in the format string must match the input text; scanning
	stops if it does not, with the return value of the function
	indicating the number of arguments scanned.

	In all the scanning functions, if an operand implements method
	Scan (that is, it implements the Scanner interface) that
	method will be used to scan the text for that operand.  Also,
	if the number of arguments scanned is less than the number of
	arguments provided, an error is returned.

	All arguments to be scanned must be either pointers to basic
	types or implementations of the Scanner interface.

	Note: Fscan etc. can read one character (rune) past the input
	they return, which means that a loop calling a scan routine
	may skip some of the input.  This is usually a problem only
	when there is no space between input values.  If the reader
	provided to Fscan implements ReadRune, that method will be used
	to read characters.  If the reader also implements UnreadRune,
	that method will be used to save the character and successive
	calls will not lose data.  To attach ReadRune and UnreadRune
	methods to a reader without that capability, use
	bufio.NewReader.
*/
package fmt
