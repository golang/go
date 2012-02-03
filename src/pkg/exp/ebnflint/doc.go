// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Ebnflint verifies that EBNF productions are consistent and grammatically correct.
It reads them from an HTML document such as the Go specification.

Grammar productions are grouped in boxes demarcated by the HTML elements
	<pre class="ebnf">
	</pre>


Usage:
	go tool ebnflint [--start production] [file]

The --start flag specifies the name of the start production for
the grammar; it defaults to "Start".

*/
package documentation
