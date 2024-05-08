These files are processed by example_test.go:TestExamples.

A .golden file is a txtar file with two sections for each example that should be
created by doc.Examples from the corresponding .go file.

One section, named EXAMPLE_NAME.Output, contains the example's output,
the value of the field Example.Output.

The other, named EXAMPLE_NAME.Play, contains the formatted code for a playable
version of the example, the value of the field Example.Play.

If a section is missing, it is treated as being empty.
Hello World
