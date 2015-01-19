// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	Package fmt implements formatted I/O with functions analogous
	to C's printf and scanf.  The format 'verbs' are derived from C's but
	are simpler.


	Printing

	The verbs:

	General:
		%v	the value in a default format
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
		%b	decimalless scientific notation with exponent a power of two,
			in the manner of strconv.FormatFloat with the 'b' format,
			e.g. -123456p-78
		%e	scientific notation, e.g. -1234.456e+78
		%E	scientific notation, e.g. -1234.456E+78
		%f	decimal point but no exponent, e.g. 123.456
		%F	synonym for %f
		%g	%e for large exponents, %f otherwise
		%G	%E for large exponents, %F otherwise
	String and slice of bytes:
		%s	the uninterpreted bytes of the string or slice
		%q	a double-quoted string safely escaped with Go syntax
		%x	base 16, lower-case, two characters per byte
		%X	base 16, upper-case, two characters per byte
	Pointer:
		%p	base 16 notation, with leading 0x

	There is no 'u' flag.  Integers are printed unsigned if they have unsigned type.
	Similarly, there is no need to specify the size of the operand (int8, int64).

	The default format for %v is:
		bool:                    %t
		int, int8 etc.:          %d
		uint, uint8 etc.:        %d, %x if printed with %#v
		float32, complex64, etc: %g
		string:                  %s
		chan:                    %p
		pointer:                 %p
	For compound objects, the elements are printed using these rules, recursively,
	laid out like this:
		struct:             {field0 field1 ...}
		array, slice:       [elem0  elem1 ...]
		maps:               map[key1:value1 key2:value2]
		pointer to above:   &{}, &[], &map[]

	Width is specified by an optional decimal number immediately preceding the verb.
	If absent, the width is whatever is necessary to represent the value.
	Precision is specified after the (optional) width by a period followed by a
	decimal number. If no period is present, a default precision is used.
	A period with no following number specifies a precision of zero.
	Examples:
		%f     default width, default precision
		%9f    width 9, default precision
		%.2f   default width, precision 2
		%9.2f  width 9, precision 2
		%9.f   width 9, precision 0

	Width and precision are measured in units of Unicode code points,
	that is, runes. (This differs from C's printf where the
	units are always measured in bytes.) Either or both of the flags
	may be replaced with the character '*', causing their values to be
	obtained from the next operand, which must be of type int.

	For most values, width is the minimum number of runes to output,
	padding the formatted form with spaces if necessary.

	For strings, byte slices and byte arrays, however, precision
	limits the length of the input to be formatted (not the size of
	the output), truncating if necessary. Normally it is measured in
	runes, but for these types when formatted with the %x or %X format
	it is measured in bytes.

	For floating-point values, width sets the minimum width of the field and
	precision sets the number of places after the decimal, if appropriate,
	except that for %g/%G it sets the total number of digits. For example,
	given 123.45 the format %6.2f prints 123.45 while %.4g prints 123.5.
	The default precision for %e and %f is 6; for %g it is the smallest
	number of digits necessary to identify the value uniquely.

	For complex numbers, the width and precision apply to the two
	components independently and the result is parenthesized, so %f applied
	to 1.2+3.4i produces (1.200000+3.400000i).

	Other flags:
		+	always print a sign for numeric values;
			guarantee ASCII-only output for %q (%+q)
		-	pad with spaces on the right rather than the left (left-justify the field)
		#	alternate format: add leading 0 for octal (%#o), 0x for hex (%#x);
			0X for hex (%#X); suppress 0x for %p (%#p);
			for %q, print a raw (backquoted) string if strconv.CanBackquote
			returns true;
			write e.g. U+0078 'x' if the character is printable for %U (%#U).
		' '	(space) leave a space for elided sign in numbers (% d);
			put spaces between bytes printing strings or slices in hex (% x, % X)
		0	pad with leading zeros rather than spaces;
			for numbers, this moves the padding after the sign

	Flags are ignored by verbs that do not expect them.
	For example there is no alternate decimal format, so %#d and %d
	behave identically.

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

	Except when printed using the verbs %T and %p, special
	formatting considerations apply for operands that implement
	certain interfaces. In order of application:

	1. If an operand implements the Formatter interface, it will
	be invoked. Formatter provides fine control of formatting.

	2. If the %v verb is used with the # flag (%#v) and the operand
	implements the GoStringer interface, that will be invoked.

	If the format (which is implicitly %v for Println etc.) is valid
	for a string (%s %q %v %x %X), the following two rules apply:

	3. If an operand implements the error interface, the Error method
	will be invoked to convert the object to a string, which will then
	be formatted as required by the verb (if any).

	4. If an operand implements method String() string, that method
	will be invoked to convert the object to a string, which will then
	be formatted as required by the verb (if any).

	For compound operands such as slices and structs, the format
	applies to the elements of each operand, recursively, not to the
	operand as a whole. Thus %q will quote each element of a slice
	of strings, and %6.2f will control formatting for each element
	of a floating-point array.

	To avoid recursion in cases such as
		type X string
		func (x X) String() string { return Sprintf("<%s>", x) }
	convert the value before recurring:
		func (x X) String() string { return Sprintf("<%s>", string(x)) }
	Infinite recursion can also be triggered by self-referential data
	structures, such as a slice that contains itself as an element, if
	that type has a String method. Such pathologies are rare, however,
	and the package does not protect against them.

	Explicit argument indexes:

	In Printf, Sprintf, and Fprintf, the default behavior is for each
	formatting verb to format successive arguments passed in the call.
	However, the notation [n] immediately before the verb indicates that the
	nth one-indexed argument is to be formatted instead. The same notation
	before a '*' for a width or precision selects the argument index holding
	the value. After processing a bracketed expression [n], subsequent verbs
	will use arguments n+1, n+2, etc. unless otherwise directed.

	For example,
		fmt.Sprintf("%[2]d %[1]d\n", 11, 22)
	will yield "22 11", while
		fmt.Sprintf("%[3]*.[2]*[1]f", 12.0, 2, 6),
	equivalent to
		fmt.Sprintf("%6.2f", 12.0),
	will yield " 12.00". Because an explicit index affects subsequent verbs,
	this notation can be used to print the same values multiple times
	by resetting the index for the first argument to be repeated:
		fmt.Sprintf("%d %d %#[1]x %#x", 16, 17)
	will yield "16 17 0x10 0x11".

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
		Invalid or invalid use of argument index: %!(BADINDEX)
			Printf("%*[2]d", 7):       %!d(BADINDEX)
			Printf("%.[2]d", 7):       %!d(BADINDEX)

	All errors begin with the string "%!" followed sometimes
	by a single character (the verb) and end with a parenthesized
	description.

	If an Error or String method triggers a panic when called by a
	print routine, the fmt package reformats the error message
	from the panic, decorating it with an indication that it came
	through the fmt package.  For example, if a String method
	calls panic("bad"), the resulting formatted message will look
	like
		%!s(PANIC=bad)

	The %!s just shows the print verb in use when the failure
	occurred.

	Scanning

	An analogous set of functions scans formatted text to yield
	values.  Scan, Scanf and Scanln read from os.Stdin; Fscan,
	Fscanf and Fscanln read from a specified io.Reader; Sscan,
	Sscanf and Sscanln read from an argument string.  Scanln,
	Fscanln and Sscanln stop scanning at a newline and require that
	the items be followed by one; Scanf, Fscanf and Sscanf require
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
		Flags # and + are not implemented.

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

	In all the scanning functions, a carriage return followed
	immediately by a newline is treated as a plain newline
	(\r\n means the same as \n).

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
