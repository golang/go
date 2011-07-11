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
Actions may not span newlines.

Actions

Here is the list of actions. "Arguments" and "pipelines" are evaluations of
data, defined in detail below.

*/
//	{{/* a comment */}}
//		A comment; discarded. Comments do not nest.
/*

	{{pipeline}}
		The default textual representation of the value of the pipeline
		is copied to the output.

	{{if pipeline}} T1 {{end}}
		If the value of the pipeline is the zero value for its type, no
		output is generated; otherwise, T1 is executed. Dot is
		unaffected.

	{{if pipeline}} T1 {{else}} T0 {{end}}
		If the value of the pipeline is the zero value for its type, T0
		is executed; otherwise, T1 is executed. Dot is unaffected.

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

	{{template argument}}
		If the value of the argument is a string, the template with that
		name is executed with nil data. If the value of arg is of type
		*Template, that template is executed.

	{{template argument pipeline}}
		If the value of the argument is a string, the template with that
		name is executed with data set to the value of the pipeline . If
		the value of arg is of type *Template, that template is
		executed.

	{{with pipeline}} T1 {{end}}
		If the value of the pipeline is the zero value for its type, no
		output is generated; otherwise, dot is set to the value of the
		pipeline and T1 is executed.

	{{with pipeline}} T1 {{else}} T0 {{end}}
		If the value of the pipeline is the zero value for its type, dot
		is unaffected and T0 is executed; otherwise, dot is set to the
		value of the pipeline and T1 is executed.

Arguments

An argument is a simple value, denoted by one of the following:

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
	- The name of a niladic method of the data, preceded by a period,
	  such as
		.Method
	  The result is the value of invoking the method with dot as the
	  receiver, dot.Method(). Such methods must have one return value (of
	  any type) or two return values, the second of which is an os.Error.
	  If it has two and the returned error is non-nil, execution terminates
	  and that error is returned to the caller as the value of Execute.
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
		The result is the value of calling the method with the arguments:
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
which has type os.Error. If that second value is present and evaluates to
non-nil, execution terminates and the error is returned to the caller of
Execute.

Variables

A pipeline may initialize a single variable to capture the result. The
initialization has syntax

	$variable := pipeline

where $variable is the name of the variable. The one exception is a pipeline in
a range action; in ranges, the variable is set to the successive elements of the
iteration.

When execution begins, $ is set to the data argument passed to Execute, that is,
to the starting value of dot.

Examples

Here are some example one-line templates demonstrating pipelines and variables.
All produce the quoted word "output":

	{{"\"output\""}}
		A string constant.
	{{`"output"`}}
		A raw string constant.
	{{printf "%q" output}}
		A function call.
	{{"output" | printf "%q"}}
		A function call whose final argument comes from the previous
		command.
	{{"put" | printf "%s%s" "out" | printf "%q"}}
		A more elaborate call.
	{{"output" | printf "%s" | printf "%q"}}
		A longer chain.
	{{$x := "output" | printf "%s" | printf "%q"}}
		An unused variables captures the output.
	{{with "output"}}{{printf "%q" .}}{{end}}
		A with action using dot.
	{{with $x := "output" | printf "%q"}}{{$x}}{{end}}
		A with action creates and uses a variable.
	{{with $x := "output"}}{{printf "%q" $x}}{{end}}
		A with action uses the variable in another action.
	{{with $x := "output"}}{{$x | printf "%q"}}{{end}}
		The same, but pipelined.

Functions

During execution functions are found in three function maps: first in the
template, then in the "template set" (described below), and finally in the
global function map. By default, no functions are defined in the template or
the set but the Funcs methods can be used to add them.

Predefined global functions are named as follows.

	and
		Returns the boolean and AND of its arguments.
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
	not
		Returns the boolean negation of its single argument.
	or
		Returns the booland OR of its arguments.
	print
		An alias for fmt.Sprint
	printf
		An alias for fmt.Sprintf
	println
		An alias for fmt.Sprintln

The boolean functions take any zero value to be false and a non-zero value to
be true.

Template sets

All templates are named by a string specified when they are created. A template
may use a template invocation to instantiate another template directly or by its
name; see the explanation of the template action above. The name of a template
is looked up in the template set active during the invocation.

If no template invocation actions occur in the template, the issue of template
sets can be ignored. If it does contain invocations, though, a set must be
defined in which to look up the names.

There are two ways to construct template sets.

The first is to use the Parse method of Set to create a set of named templates
by reading a single string defining multiple templates. The syntax of the
definitions is to surround each template declaration with a define and end
action; those actions are discarded after parsing.

The define action names the template being created by providing a string
constant. Here is a simple example of input to Set.Parse:

	`{{define "T1"}} definition of template T1 {{end}}
	{{define "T2"}} definition of template T2 {{end}}
	{{define "T3"}} {{template "T1"}} {{template "T2"}} {{end}}`

This defines two templates, T1 and T2, and a third T3 that invokes the other two
when it is executed.

The second way to build a template set is to use the Add method of Set to bind
a template to a set. A template may be bound to multiple sets.

When templates are executed via Template.Execute, no set is defined and so no
template invocations are possible. The method Template.ExecuteInSet provides a
way to specify a template set when executing a template directly.

A more direct technique is to use Set.Execute, which executes a named template
from the set and provides the context for looking up templates in template
invocations. To invoke our example above, we might write,

	err := set.Execute("T3", os.Stdout, "no data needed")
	if err != nil {
		log.Fatalf("execution failed: %s", err)
	}
*/
package template
