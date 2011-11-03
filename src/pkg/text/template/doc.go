// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package template implements data-driven templates for generating textual output
such as HTML.

Templates are executed by applying them to a data structure. Annotations in the
template refer to elements of the data structure (typically a field of a struct
or a key in a map) to control execution and derive values to be displayed.
Execution of the template walks the structure and sets the cursor, represented
by a period '.' and called "dot", to the value at the current location in the
structure as execution proceeds.

The input text for a template is UTF-8-encoded text in any format.
"Actions"--data evaluations or control structures--are delimited by
"{{" and "}}"; all text outside actions is copied to the output unchanged.
Actions may not span newlines, although comments can.

Once constructed, templates and template sets can be executed safely in
parallel.

Actions

Here is the list of actions. "Arguments" and "pipelines" are evaluations of
data, defined in detail below.

*/
//	{{/* a comment */}}
//		A comment; discarded. May contain newlines.
//		Comments do not nest.
/*

	{{pipeline}}
		The default textual representation of the value of the pipeline
		is copied to the output.

	{{if pipeline}} T1 {{end}}
		If the value of the pipeline is empty, no output is generated;
		otherwise, T1 is executed.  The empty values are false, 0, any
		nil pointer or interface value, and any array, slice, map, or
		string of length zero.
		Dot is unaffected.

	{{if pipeline}} T1 {{else}} T0 {{end}}
		If the value of the pipeline is empty, T0 is executed;
		otherwise, T1 is executed.  Dot is unaffected.

	{{range pipeline}} T1 {{end}}
		The value of the pipeline must be an array, slice, or map. If
		the value of the pipeline has length zero, nothing is output;
		otherwise, dot is set to the successive elements of the array,
		slice, or map and T1 is executed.

	{{range pipeline}} T1 {{else}} T0 {{end}}
		The value of the pipeline must be an array, slice, or map. If
		the value of the pipeline has length zero, dot is unaffected and
		T0 is executed; otherwise, dot is set to the successive elements
		of the array, slice, or map and T1 is executed.

	{{template "name"}}
		The template with the specified name is executed with nil data.

	{{template "name" pipeline}}
		The template with the specified name is executed with dot set
		to the value of the pipeline.

	{{with pipeline}} T1 {{end}}
		If the value of the pipeline is empty, no output is generated;
		otherwise, dot is set to the value of the pipeline and T1 is
		executed.

	{{with pipeline}} T1 {{else}} T0 {{end}}
		If the value of the pipeline is empty, dot is unaffected and T0
		is executed; otherwise, dot is set to the value of the pipeline
		and T1 is executed.

Arguments

An argument is a simple value, denoted by one of the following.

	- A boolean, string, character, integer, floating-point, imaginary
	  or complex constant in Go syntax. These behave like Go's untyped
	  constants, although raw strings may not span newlines.
	- The character '.' (period):
		.
	  The result is the value of dot.
	- A variable name, which is a (possibly empty) alphanumeric string
	  preceded by a dollar sign, such as
		$piOver2
	  or
		$
	  The result is the value of the variable.
	  Variables are described below.
	- The name of a field of the data, which must be a struct, preceded
	  by a period, such as
		.Field
	  The result is the value of the field. Field invocations may be
	  chained:
	    .Field1.Field2
	  Fields can also be evaluated on variables, including chaining:
	    $x.Field1.Field2
	- The name of a key of the data, which must be a map, preceded
	  by a period, such as
		.Key
	  The result is the map element value indexed by the key.
	  Key invocations may be chained and combined with fields to any
	  depth:
	    .Field1.Key1.Field2.Key2
	  Although the key must be an alphanumeric identifier, unlike with
	  field names they do not need to start with an upper case letter.
	  Keys can also be evaluated on variables, including chaining:
	    $x.key1.key2
	- The name of a niladic method of the data, preceded by a period,
	  such as
		.Method
	  The result is the value of invoking the method with dot as the
	  receiver, dot.Method(). Such a method must have one return value (of
	  any type) or two return values, the second of which is an error.
	  If it has two and the returned error is non-nil, execution terminates
	  and an error is returned to the caller as the value of Execute.
	  Method invocations may be chained and combined with fields and keys
	  to any depth:
	    .Field1.Key1.Method1.Field2.Key2.Method2
	  Methods can also be evaluated on variables, including chaining:
	    $x.Method1.Field
	- The name of a niladic function, such as
		fun
	  The result is the value of invoking the function, fun(). The return
	  types and values behave as in methods. Functions and function
	  names are described below.

Arguments may evaluate to any type; if they are pointers the implementation
automatically indirects to the base type when required.

A pipeline is a possibly chained sequence of "commands". A command is a simple
value (argument) or a function or method call, possibly with multiple arguments:

	Argument
		The result is the value of evaluating the argument.
	.Method [Argument...]
		The method can be alone or the last element of a chain but,
		unlike methods in the middle of a chain, it can take arguments.
		The result is the value of calling the method with the
		arguments:
			dot.Method(Argument1, etc.)
	functionName [Argument...]
		The result is the value of calling the function associated
		with the name:
			function(Argument1, etc.)
		Functions and function names are described below.

Pipelines

A pipeline may be "chained" by separating a sequence of commands with pipeline
characters '|'. In a chained pipeline, the result of the each command is
passed as the last argument of the following command. The output of the final
command in the pipeline is the value of the pipeline.

The output of a command will be either one value or two values, the second of
which has type error. If that second value is present and evaluates to
non-nil, execution terminates and the error is returned to the caller of
Execute.

Variables

A pipeline inside an action may initialize a variable to capture the result.
The initialization has syntax

	$variable := pipeline

where $variable is the name of the variable. An action that declares a
variable produces no output.

If a "range" action initializes a variable, the variable is set to the
successive elements of the iteration.  Also, a "range" may declare two
variables, separated by a comma:

	$index, $element := pipeline

in which case $index and $element are set to the successive values of the
array/slice index or map key and element, respectively.  Note that if there is
only one variable, it is assigned the element; this is opposite to the
convention in Go range clauses.

A variable's scope extends to the "end" action of the control structure ("if",
"with", or "range") in which it is declared, or to the end of the template if
there is no such control structure.  A template invocation does not inherit
variables from the point of its invocation.

When execution begins, $ is set to the data argument passed to Execute, that is,
to the starting value of dot.

Examples

Here are some example one-line templates demonstrating pipelines and variables.
All produce the quoted word "output":

	{{"\"output\""}}
		A string constant.
	{{`"output"`}}
		A raw string constant.
	{{printf "%q" "output"}}
		A function call.
	{{"output" | printf "%q"}}
		A function call whose final argument comes from the previous
		command.
	{{"put" | printf "%s%s" "out" | printf "%q"}}
		A more elaborate call.
	{{"output" | printf "%s" | printf "%q"}}
		A longer chain.
	{{with "output"}}{{printf "%q" .}}{{end}}
		A with action using dot.
	{{with $x := "output" | printf "%q"}}{{$x}}{{end}}
		A with action that creates and uses a variable.
	{{with $x := "output"}}{{printf "%q" $x}}{{end}}
		A with action that uses the variable in another action.
	{{with $x := "output"}}{{$x | printf "%q"}}{{end}}
		The same, but pipelined.

Functions

During execution functions are found in three function maps: first in the
template, then in the "template set" (described below), and finally in the
global function map. By default, no functions are defined in the template or
the set but the Funcs methods can be used to add them.

Predefined global functions are named as follows.

	and
		Returns the boolean AND of its arguments by returning the
		first empty argument or the last argument, that is,
		"and x y" behaves as "if x then y else x". All the
		arguments are evaluated.
	html
		Returns the escaped HTML equivalent of the textual
		representation of its arguments.
	index
		Returns the result of indexing its first argument by the
		following arguments. Thus "index x 1 2 3" is, in Go syntax,
		x[1][2][3]. Each indexed item must be a map, slice, or array.
	js
		Returns the escaped JavaScript equivalent of the textual
		representation of its arguments.
	len
		Returns the integer length of its argument.
	not
		Returns the boolean negation of its single argument.
	or
		Returns the boolean OR of its arguments by returning the
		first non-empty argument or the last argument, that is,
		"or x y" behaves as "if x then x else y". All the
		arguments are evaluated.
	print
		An alias for fmt.Sprint
	printf
		An alias for fmt.Sprintf
	println
		An alias for fmt.Sprintln
	urlquery
		Returns the escaped value of the textual representation of
		its arguments in a form suitable for embedding in a URL query.

The boolean functions take any zero value to be false and a non-zero value to
be true.

Template sets

Each template is named by a string specified when it is created.  A template may
use a template invocation to instantiate another template directly or by its
name; see the explanation of the template action above. The name is looked up
in the template set associated with the template.

If no template invocation actions occur in the template, the issue of template
sets can be ignored.  If it does contain invocations, though, the template
containing the invocations must be part of a template set in which to look up
the names.

There are two ways to construct template sets.

The first is to use a Set's Parse method to create a set of named templates from
a single input defining multiple templates.  The syntax of the definitions is to
surround each template declaration with a define and end action.

The define action names the template being created by providing a string
constant. Here is a simple example of input to Set.Parse:

	`{{define "T1"}} definition of template T1 {{end}}
	{{define "T2"}} definition of template T2 {{end}}
	{{define "T3"}} {{template "T1"}} {{template "T2"}} {{end}}`

This defines two templates, T1 and T2, and a third T3 that invokes the other two
when it is executed.

The second way to build a template set is to use Set's Add method to add a
parsed template to a set.  A template may be bound to at most one set.  If it's
necessary to have a template in multiple sets, the template definition must be
parsed multiple times to create distinct *Template values.

Set.Parse may be called multiple times on different inputs to construct the set.
Two sets may therefore be constructed with a common base set of templates plus,
through a second Parse call each, specializations for some elements.

A template may be executed directly or through Set.Execute, which executes a
named template from the set.  To invoke our example above, we might write,

	err := set.Execute(os.Stdout, "T3", "no data needed")
	if err != nil {
		log.Fatalf("execution failed: %s", err)
	}
*/
package template
