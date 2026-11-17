# Preserved patch context

The staged whitespace check reports the literal one-space context line at
line 4 of `arm64-recipe-v2/arm64-recipe.patch`. That line is part of the
unchanged historical unified diff, whose exact bytes are covered by the V2
manifest. It is not whitespace added to executable source.

The complete staged whitespace check excluding that one archived patch passes.
The archived patch is preserved without normalization.
